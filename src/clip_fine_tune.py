import json
import multiprocessing
import clip
import pandas as pd
import torch
import torch.nn.functional as F

from comet_ml import Experiment
from argparse import ArgumentParser
from pathlib import Path
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, squarepad_transform, targetpad_transform, square_transform, CPRDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, save_model, element_wise_sum, device
from validate import compute_cpr_val_metrics


def clip_finetune_cpr(dataset: str, experiment_name: str, num_epochs: int, clip_model_name: str, learning_rate: float, 
                      batch_size: int, validation_frequency: int, transform: str, save_training: bool, encoder: str,  
                      train_split: str, val_split: str, mllm_mode: str, use_cycle_loss: bool, cycle_loss_weight: float, 
                      use_auto_sents: int, use_human_sents: int,
                      **kwargs):
    """
    Fine-tune CLIP on a CPR dataset using "image-text element-wise sum" as the combining function
    :param dataset: CPR dataset to use, should be in ['posefix', 'aist', 'posefix']
    :param experiment_name: name of the experiment
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4", "ViT-B/32"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']

    :param train_split: which train split you want to use
    :param val_split: which validation split you want to use
    :param mllm mode, e.g., "mllm-reverse-mirror-filter_3"
    :param use_cycle_loss: Whether to use cycle loss
    :param cycle_loss_weight: Weight for cycle loss
    :param use_auto_sents: How many auto sentences to use
    :param use_human_sents: How many human sentences to use
    :param kwargs: 
        - if you use the `targetpad` transform you should provide `target_ratio` as kwargs
    """

    # Set up the training path
    model_name_str = clip_model_name.replace('/', '-')
    training_path: Path = Path(base_path / f"models/{model_name_str}/clip_finetuned_on_{dataset}_{experiment_name}")        
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # Load the CLIP model
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    # Set up trainable parameters based on the encoder choice
    if encoder == 'text':
        print('Only the CLIP text encoder will be fine-tuned')
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'image':
        print('Only the CLIP image encoder will be fine-tuned')
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in clip_model.visual.parameters():
            param.requires_grad = True
    elif encoder == 'both':
        print('Both CLIP encoders will be fine-tuned')
    else:
        raise ValueError("encoder parameter should be in ['text', 'image', both']")

    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution

    # Set up the preprocess transform
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

    # Define the validation datasets
    relative_val_dataset = CPRDataset(val_split, dataset, 'relative', preprocess)
    classic_val_dataset = CPRDataset(val_split, dataset, 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over the epochs
    if encoder == 'text':
        val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the train dataset and the combining function
    relative_train_dataset = CPRDataset(train_split, dataset, 'relative', preprocess, mllm_mode, use_cycle_loss, use_auto_sents, use_human_sents)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            if not use_cycle_loss:
                for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                    images_in_batch = reference_images.size(0)
                    step = len(train_bar) * epoch + idx

                    optimizer.zero_grad()

                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)

                    # Extract the features, compute the logits and the loss
                    with torch.cuda.amp.autocast():
                        reference_features = clip_model.encode_image(reference_images)
                        text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device,
                                                                                                non_blocking=True)
                        text_features = clip_model.encode_text(text_inputs)

                        target_features = F.normalize(clip_model.encode_image(target_images), dim=-1)
                        predicted_features = combining_function(reference_features, text_features)

                        logits = 100 * predicted_features @ target_features.T

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
                for idx, (reference_images, target_images, captions, reverse_captions) in enumerate(train_bar):
                    images_in_batch = reference_images.size(0)
                    step = len(train_bar) * epoch + idx

                    optimizer.zero_grad()

                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)

                    # Extract the features, compute the logits and the loss
                    with torch.cuda.amp.autocast():
                        # ref imgs
                        reference_features = clip_model.encode_image(reference_images)
                        norm_reference_features = F.normalize(reference_features, dim=-1)

                        # ref->tgt texts
                        text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device, non_blocking=True)
                        text_features = clip_model.encode_text(text_inputs)

                        # tgt imgs
                        target_features = clip_model.encode_image(target_images)
                        norm_target_features = F.normalize(target_features, dim=-1)

                        # tgt->ref texts
                        text_inputs_reverse = clip.tokenize(reverse_captions, context_length=77, truncate=True).to(device, non_blocking=True)
                        text_features_reverse = clip_model.encode_text(text_inputs_reverse)

                        # ref imgs + ref->tgt texts = tgt imgs
                        predicted_features = combining_function(reference_features, text_features)
                        logits = 100 * predicted_features @ norm_target_features.T

                        # tgt imgs + tgt->ref texts = ref imgs
                        predicted_features_reverse = combining_function(predicted_features, text_features_reverse)
                        logits_reverse = 100 * predicted_features_reverse @ norm_reference_features.T                

                        # reconstruction loss + cycle loss
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
            with experiment.validate():
                if encoder != 'text':
                    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

                results = compute_cpr_val_metrics(relative_val_dataset, clip_model, val_index_features, val_index_names, combining_function)
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

                if save_training and (epoch == (num_epochs - 1)):
                    save_model(f'tuned_clip_{epoch}', epoch, clip_model, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be 'FixMyPose', 'AIST', 'PoseFix'")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--encoder", default='both', type=str, help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true', help="Whether save the training model")

    # AutoComPose specific arguments
    parser.add_argument("--train-split", default="train", type=str, help="Train split to use")
    parser.add_argument("--val-split", default="test", type=str, help="Valid split to use")
    parser.add_argument("--mllm-mode", default=None, type=str, help="MLLM mode to use, e.g., 'mllm-mirror-reverse_5'")
    parser.add_argument("--use-cycle-loss", dest="use_cycle_loss", action='store_true', help="Whether to use cycle loss")
    parser.add_argument("--cycle-loss-weight", default=0.5, type=float, help="Weight for cycle loss")
    parser.add_argument("--use-auto-sents", default=0, type=int, help="How many auto sentences to use")
    parser.add_argument("--use-human-sents", default=0, type=int, help="How many human sentences to use")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fixmypose', 'aist', 'posefix']:
        raise ValueError("Dataset should be 'FixMyPose', 'AIST', 'PoseFix'")

    training_hyper_params = {
        "dataset": args.dataset.lower(),
        "experiment_name": args.experiment_name,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "encoder": args.encoder,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "mllm_mode": args.mllm_mode,
        "use_cycle_loss": args.use_cycle_loss,
        "cycle_loss_weight": args.cycle_loss_weight,
        "use_auto_sents": args.use_auto_sents,
        "use_human_sents": args.use_human_sents,
    }

    # Set up the experiment logging
    print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
        disabled=True
    )
    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    # Call the fine-tuning function
    clip_finetune_cpr(**training_hyper_params)