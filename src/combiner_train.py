import json
import clip
import pandas as pd
import torch

from comet_ml import Experiment
from argparse import ArgumentParser
from pathlib import Path
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, squarepad_transform, targetpad_transform, square_transform, CPRDataset
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, extract_index_features, device
from validate import compute_cpr_val_metrics


def combiner_training_cpr(dataset: str, experiment_name: str, projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str, save_training: bool,
                          train_split: str, val_split: str, mllm_mode: str, use_cycle_loss: bool, cycle_loss_weight: float, 
                          use_auto_sents: int, use_human_sents: int, 
                          **kwargs):
    """
    Train the Combiner on a CPR dataset keeping frozen the CLIP model
    :param dataset: CPR dataset to use, should be in ['posefix', 'aist', 'posefix']
    :param experiment_name: name of the experiment
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
    :param save_training: when True save the weights of the Combiner network

    :param train_split: which train split you want to use
    :param val_split: which validation split you want to use
    :param mllm mode, e.g., "mllm-reverse-mirror-filter_3"
    :param use_cycle_loss: Whether to use cycle loss
    :param cycle_loss_weight: Weight for cycle loss
    :param use_auto_sents: How many auto sentences to use
    :param use_human_sents: How many human sentences to use
    :param kwargs: 
        - if you use the `targetpad` transform you should prove `target_ratio` as kwarg
        - if you want to load a fine-tuned version of clip you should provide `clip_model_path` as kwarg
        - if you want to load a pre-trained Combiner model you should provide `combiner_model_path` as kwarg
    """

    # Set up the training path
    model_name_str = clip_model_name.replace('/', '-')
    training_path: Path = Path(base_path / f"models/{model_name_str}/combiner_trained_on_{dataset}_{experiment_name}") 
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # Load the CLIP model
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "square":
        preprocess = square_transform(input_dim)
        print('Square preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Load the CLIP model if a path is provided
    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    clip_model = clip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = CPRDataset(val_split, dataset, 'relative', preprocess)
    classic_val_dataset = CPRDataset(val_split, dataset, 'classic', preprocess)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the combiner
    if use_cycle_loss:
        combiner = Combiner(feature_dim, projection_dim, hidden_dim, True).to(device, non_blocking=True)
    else:
        combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)

    # Define the train dataset
    relative_train_dataset = CPRDataset(train_split, dataset, 'relative', preprocess, mllm_mode, use_cycle_loss, use_auto_sents, use_human_sents)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)
    
    # Load the pre-trained Combiner model if a path is provided
    if kwargs.get("combiner_model_path"):
        print('Trying to load the pre-trained Combiner model')
        combiner_model_path = kwargs["combiner_model_path"]
        state_dict = torch.load(combiner_model_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        print('Combiner model loaded successfully')

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            if not use_cycle_loss:
                for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                    images_in_batch = reference_images.size(0)
                    step = len(train_bar) * epoch + idx

                    optimizer.zero_grad()

                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)
                    text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)

                    # Extract the features with CLIP
                    with torch.no_grad():
                        reference_images_list = torch.split(reference_images, clip_bs)
                        reference_features = torch.vstack(
                            [clip_model.encode_image(mini_batch).float() for mini_batch in reference_images_list])
                        target_images_list = torch.split(target_images, clip_bs)
                        target_features = torch.vstack(
                            [clip_model.encode_image(mini_batch).float() for mini_batch in target_images_list])

                        text_inputs_list = torch.split(text_inputs, clip_bs)
                        text_features = torch.vstack(
                            [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list])

                    # Compute the logits and loss
                    with torch.cuda.amp.autocast():
                        logits = combiner(reference_features, text_features, target_features)
                        ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                        loss = crossentropy_criterion(logits, ground_truth)

                    # Backpropagate and update the weights
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                    update_train_running_results(train_running_results, loss, images_in_batch)
                    set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
            # Use cycle loss
            else:
                for idx, (reference_images, target_images, captions, reverse_captions) in enumerate(train_bar):  # Load a batch of triplets
                    images_in_batch = reference_images.size(0)
                    step = len(train_bar) * epoch + idx

                    optimizer.zero_grad()

                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)
                    text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)
                    text_inputs_reverse = clip.tokenize(reverse_captions, truncate=True).to(device, non_blocking=True)

                    # Extract the features with CLIP
                    with torch.no_grad():
                        reference_images_list = torch.split(reference_images, clip_bs)
                        reference_features = torch.vstack(
                            [clip_model.encode_image(mini_batch).float() for mini_batch in reference_images_list])
                        target_images_list = torch.split(target_images, clip_bs)
                        target_features = torch.vstack(
                            [clip_model.encode_image(mini_batch).float() for mini_batch in target_images_list])

                        text_inputs_list = torch.split(text_inputs, clip_bs)
                        text_features = torch.vstack(
                            [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list])
                        text_inputs_list_reverse = torch.split(text_inputs_reverse, clip_bs)
                        text_features_reverse = torch.vstack(
                            [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list_reverse])

                    # Compute the logits and loss
                    with torch.cuda.amp.autocast():
                        logits, combined_features = combiner(reference_features, text_features, target_features)
                        logits_reverse, _ = combiner(combined_features, text_features_reverse, reference_features)
                        ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                        loss = (1.0 - cycle_loss_weight)*crossentropy_criterion(logits, ground_truth) + cycle_loss_weight*crossentropy_criterion(logits_reverse, ground_truth)

                    # Backpropagate and update the weights
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                    update_train_running_results(train_running_results, loss, images_in_batch)
                    set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)                

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if (epoch % validation_frequency == 0) or (epoch == (num_epochs - 1)):
            clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()

                # Compute and log validation metrics
                results = compute_cpr_val_metrics(relative_val_dataset, clip_model, val_index_features, val_index_names, combiner.combine_features)
                group_recall_at1, group_recall_at3, group_recall_at5, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at3': group_recall_at3,
                    'group_recall_at5': group_recall_at5,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                }

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                # Save model
                if save_training and (epoch == (num_epochs - 1)):
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be 'FixMyPose', 'AIST', 'PoseFix'")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--combiner-model-path", type=str, help="Path to the pre-trained Combiner model")

    # AutoCompose specific arguments
    parser.add_argument("--train-split", default="train", type=str, help="Train split to use")
    parser.add_argument("--val-split", default="test", type=str, help="Valid split to use")
    parser.add_argument("--mllm-mode", default=None, type=str, help="MLLM mode to use, e.g., 'mllm-mirror-reverse_5'")
    parser.add_argument("--use-cycle-loss", dest="use_cycle_loss", action='store_true', help="Whether to use cycle loss")
    parser.add_argument("--cycle-loss-weight", default=0.5, type=float, help="Weight for cycle loss")
    parser.add_argument("--use-auto-sents", default=0, type=int, help="How many auto sentences to use")
    parser.add_argument("--use-human-sents", default=0, type=int, help="How many human sentences to use")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fixmypose', 'aist', 'posefix']:
        raise ValueError("Dataset should be 'CIRR', 'FashionIQ', 'FixMyPose', 'AIST', 'PoseFix'")

    training_hyper_params = {
        "dataset": args.dataset.lower(),
        "experiment_name": args.experiment_name,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "target_ratio": args.target_ratio,
        "transform": args.transform,
        "save_training": args.save_training,
        "combiner_model_path": args.combiner_model_path,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "mllm_mode": args.mllm_mode,
        "use_cycle_loss": args.use_cycle_loss,
        "cycle_loss_weight": args.cycle_loss_weight,
        "use_auto_sents": args.use_auto_sents,
        "use_human_sents": args.use_human_sents,
    }

    print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
        disabled=True
    )
    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    # Call the combiner training function
    combiner_training_cpr(**training_hyper_params)
