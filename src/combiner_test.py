import json
import clip
import numpy as np
import pandas as pd
import torch

from comet_ml import Experiment
from argparse import ArgumentParser
from pathlib import Path
from data_utils import base_path, squarepad_transform, targetpad_transform, square_transform, CPRDataset
from combiner import Combiner
from utils import extract_index_features, device
from validate import compute_cpr_val_metrics


def combiner_test_cpr(dataset: str, experiment_name: str, projection_dim: int, hidden_dim: int, clip_model_name: str, transform: str,
                      val_split: str, **kwargs):
    """
    Test the Combiner on a CPR dataset
    :param dataset: CPR dataset to use, should be in ['posefix', 'aist', 'posefix']
    :param experiment_name: name of the experiment
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']
    :param val_split: which validation split you want to use
    :param kwargs: 
        - if you use the `targetpad` transform you should prove `target_ratio` as kwarg
        - if you want to load a fine-tuned version of clip you should provide `clip_model_path` as kwarg
        - if you want to load a pre-trained Combiner model you should provide `combiner_model_path` as kwarg
        - if you want to keep selection results, you should provide `keep_selection` as kwarg
    """

    # Set up the training path
    model_name_str = clip_model_name.replace('/', '-')
    training_path: Path = Path(base_path / f"models/{model_name_str}/combiner_trained_on_{dataset}_{experiment_name}") 
        
    # whether to keep selection results
    keep_selection = kwargs.get("keep_selection")

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

    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')
    else:
        raise ValueError("no clip model path provided.")

    clip_model = clip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = CPRDataset(val_split, dataset, 'relative', preprocess)
    classic_val_dataset = CPRDataset(val_split, dataset, 'classic', preprocess)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    
    if kwargs.get("combiner_model_path"):
        print('Trying to load the pre-trained Combiner model')
        combiner_model_path = kwargs["combiner_model_path"]
        state_dict = torch.load(combiner_model_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        print('Combiner model loaded successfully')
    else:
        raise ValueError("no combiner model path provided.")

    # Define dataframes for CSV logging
    validation_log_frame = pd.DataFrame()

    with experiment.validate():
        combiner.eval()

        # Compute and log validation metrics
        results = compute_cpr_val_metrics(relative_val_dataset, clip_model, val_index_features, val_index_names, combiner.combine_features, keep_selection)
        
        if keep_selection > 0:
            group_recall_at1, group_recall_at3, group_recall_at5, recall_at1, recall_at5, recall_at10, recall_at50, ref_names, tgt_names, sorted_index_names, labels = results
        else:
            group_recall_at1, group_recall_at3, group_recall_at5, recall_at1, recall_at5, recall_at10, recall_at50 = results

        if keep_selection > 0:
            k = keep_selection

            # number of test images = N
            # shape = [N] (1 or 0)
            labels_arr = torch.sum(labels[:, :k], dim=1).numpy()

            # number of correct images = n
            selected_ref_names_arr = np.array(ref_names)[labels_arr == 1].reshape(-1, 1)        # (n, 1)
            selected_tgt_names_arr = np.array(tgt_names)[labels_arr == 1].reshape(-1, 1)        # (n, 1)
            selected_sorted_names_arr = np.array(sorted_index_names[:, :k])[labels_arr == 1]    # (n, k)

            combined = np.concatenate((selected_ref_names_arr, selected_tgt_names_arr, selected_sorted_names_arr), axis=1)  # (n, 2 + k)
            column_names = ["reference", "target"] + [f"selection {i}" for i in range(1, k+1)]
            df = pd.DataFrame(combined, columns=column_names)
            df.to_csv(str(training_path / f'top{k}_correct_selection.csv'), index=False)

            # since each image has two annotations
            ref_names_arr = np.array(ref_names).reshape(-1, 1)      # (N, 1)
            tgt_names_arr = np.array(tgt_names).reshape(-1, 1)      # (N, 1)
            sorted_names_arr = np.array(sorted_index_names[:, :k])  # (N, k)

            combined_all = np.concatenate((ref_names_arr, tgt_names_arr, sorted_names_arr), axis=1) # (N, 2 + k)
            df = pd.DataFrame(combined_all, columns=column_names)
            df.to_csv(str(training_path / f'top{k}_all_selection.csv'), index=False)  

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
            epoch=0
        )

        # Validation CSV logging
        log_dict = {'epoch': 0}
        log_dict.update(results_dict)
        validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
        validation_log_frame.to_csv(str(training_path / 'validation_metrics_test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be 'FixMyPose', 'AIST', 'PoseFix'")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--val-split", default="test", type=str, help="Valid split to use")
    parser.add_argument("--combiner-model-path", type=str, help="Path to the pre-trained Combiner model")
    parser.add_argument("--keep-selection", default=50, type=int, help="Save topk selection results (0 to disable)")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fixmypose', 'aist', 'posefix']:
        raise ValueError("Dataset should be 'FixMyPose', 'AIST', 'PoseFix'")

    training_hyper_params = {
        "dataset": args.dataset.lower(),
        "experiment_name": args.experiment_name,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path, #
        "transform": args.transform,
        "target_ratio": args.target_ratio, #
        "val_split": args.val_split,
        "combiner_model_path": args.combiner_model_path, #
        "keep_selection": args.keep_selection, #
    }

    print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
        disabled=True
    )

    # Call the test function
    combiner_test_cpr(**training_hyper_params)
