#!/bin/bash

set -ex

### General settings ###
transform="squarepad"
train_split="train"
val_split="test"
keep_selection=50

# CLIP #
clip_epochs=50
clip_enc="text"
clip_lr=2e-6
clip_bs=128
clip_val_f=10

# Combiner #
comb_proj=2560
comb_hidden=5120
comb_epochs=100
comb_lr=2e-5
comb_clip_bs=32
comb_val_f=20
comb_bs=512

### Other settings ###
dataset=("fixmypose" "aist" "posefix")   # Options: fixmypose, aist, and posefix
clip_names=("RN50")     # Options: RN50, RN101, ViT-B/32, and ViT-B/16

for dataset in "${dataset[@]}"; do
    for clip_name in "${clip_names[@]}"; do
        #############
        ### Human ###
        #############

        # Use Human-annotated Pose Transition Descriptions
        # n_human_sents=1     # Options: 1 for fixmypose; not supported for aist; 1-3 for posefix
        # exp_name="human-${n_human_sents}"

        # CLIP Fine-Tuning
        # python src/clip_fine_tune.py --dataset ${dataset} --experiment-name ${exp_name} --num-epochs ${clip_epochs} --clip-model-name ${clip_name} --encoder ${clip_enc} --learning-rate ${clip_lr} --batch-size ${clip_bs} --validation-frequency ${clip_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-human-sents ${n_human_sents}

        # CLIP Evaluation
        # python src/clip_test.py --dataset ${dataset} --experiment-name ${exp_name} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --keep-selection ${keep_selection}

        # Combiner Training
        # python src/combiner_train.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --num-epochs ${comb_epochs} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --combiner-lr ${comb_lr} --batch-size ${comb_bs} --clip-bs ${comb_clip_bs} --validation-frequency ${comb_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-human-sents ${n_human_sents}

        # Combiner Evaluation
        # python src/combiner_test.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --combiner-model-path ./models/${clip_name//\//-}/combiner_trained_on_${dataset}_${exp_name}/saved_models/combiner_$((comb_epochs - 1)).pt --keep-selection ${keep_selection}

        ############
        ### Auto ###
        ############

        # Use PoseFix-generated Pose Transition Descriptions (Rule-based)
        # n_auto_sents=3      # Options: 1-3 for posefix; not supported for others
        # exp_name="auto-${n_auto_sents}"

        # CLIP Fine-Tuning
        # python src/clip_fine_tune.py --dataset ${dataset} --experiment-name ${exp_name} --num-epochs ${clip_epochs} --clip-model-name ${clip_name} --encoder ${clip_enc} --learning-rate ${clip_lr} --batch-size ${clip_bs} --validation-frequency ${clip_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-auto-sents ${n_auto_sents}

        # CLIP Evaluation
        # python src/clip_test.py --dataset ${dataset} --experiment-name ${exp_name} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --keep-selection ${keep_selection}

        # Combiner Training
        # python src/combiner_train.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --num-epochs ${comb_epochs} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --combiner-lr ${comb_lr} --batch-size ${comb_bs} --clip-bs ${comb_clip_bs} --validation-frequency ${comb_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-auto-sents ${n_auto_sents}

        # Combiner Evaluation
        # python src/combiner_test.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --combiner-model-path ./models/${clip_name//\//-}/combiner_trained_on_${dataset}_${exp_name}/saved_models/combiner_$((comb_epochs - 1)).pt --keep-selection ${keep_selection}

        #####################
        ### Auto + Cyclic ###
        #####################

        # Use Forward and Backward PoseFix-generated Pose Transition Descriptions (Rule-based)
        # n_auto_sents=3      # Options: 1-3 for posefix; not supported for others
        # exp_name="auto-${n_auto_sents}-cycle"

        # CLIP Fine-Tuning
        # python src/clip_fine_tune.py --dataset ${dataset} --experiment-name ${exp_name} --num-epochs ${clip_epochs} --clip-model-name ${clip_name} --encoder ${clip_enc} --learning-rate ${clip_lr} --batch-size ${clip_bs} --validation-frequency ${clip_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-cycle-loss --use-auto-sents ${n_auto_sents}

        # CLIP Evaluation
        # python src/clip_test.py --dataset ${dataset} --experiment-name ${exp_name} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --keep-selection ${keep_selection}

        # Combiner Training
        # python src/combiner_train.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --num-epochs ${comb_epochs} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --combiner-lr ${comb_lr} --batch-size ${comb_bs} --clip-bs ${comb_clip_bs} --validation-frequency ${comb_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --use-cycle-loss --use-auto-sents ${n_auto_sents}

        # Combiner Evaluation
        # python src/combiner_test.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --combiner-model-path ./models/${clip_name//\//-}/combiner_trained_on_${dataset}_${exp_name}/saved_models/combiner_$((comb_epochs - 1)).pt --keep-selection ${keep_selection}

        ############
        ### MLLM ###
        ############

        # Use AutoComPose-generated Pose Transition Descriptions
        # modes=("mllm-mirror-reverse_3")     # Options: mllm_k, mllm-mirror_k, mllm-reverse_k, and mllm-mirror-reverse_k; k: 1, 3, or 5
        # for mode in "${modes[@]}"; do
            # CLIP Fine-Tuning
            # python src/clip_fine_tune.py --dataset ${dataset} --experiment-name ${mode} --num-epochs ${clip_epochs} --clip-model-name ${clip_name} --encoder ${clip_enc} --learning-rate ${clip_lr} --batch-size ${clip_bs} --validation-frequency ${clip_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --mllm-mode ${mode}

            # CLIP Evaluation
            # python src/clip_test.py --dataset ${dataset} --experiment-name ${mode} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${mode}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --keep-selection ${keep_selection}

            # Combiner Training
            # python src/combiner_train.py --dataset ${dataset} --experiment-name ${mode} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --num-epochs ${comb_epochs} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${mode}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --combiner-lr ${comb_lr} --batch-size ${comb_bs} --clip-bs ${comb_clip_bs} --validation-frequency ${comb_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --mllm-mode ${mode}

            # Combiner Evaluation
            # python src/combiner_test.py --dataset ${dataset} --experiment-name ${mode} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${mode}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --combiner-model-path ./models/${clip_name//\//-}/combiner_trained_on_${dataset}_${mode}/saved_models/combiner_$((comb_epochs - 1)).pt --keep-selection ${keep_selection}
        # done

        #####################
        ### MLLM + Cyclic ###
        #####################

        # Use AutoComPose-generated Pose Transition Descriptions (w/ Cyclic-Training)
        modes=("mllm-mirror-reverse_3")     # Options: mllm-reverse_k and mllm-mirror-reverse_k; k: 1, 3, or 5
        for mode in "${modes[@]}"; do
            exp_name="${mode}-cycle"
            # CLIP Fine-Tuning
            # python src/clip_fine_tune.py --dataset ${dataset} --experiment-name ${exp_name} --num-epochs ${clip_epochs} --clip-model-name ${clip_name} --encoder ${clip_enc} --learning-rate ${clip_lr} --batch-size ${clip_bs} --validation-frequency ${clip_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --mllm-mode ${mode} --use-cycle-loss

            # CLIP Evaluation
            # python src/clip_test.py --dataset ${dataset} --experiment-name ${exp_name} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --keep-selection ${keep_selection}

            # Combiner Training
            # python src/combiner_train.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --num-epochs ${comb_epochs} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --combiner-lr ${comb_lr} --batch-size ${comb_bs} --clip-bs ${comb_clip_bs} --validation-frequency ${comb_val_f} --transform ${transform} --save-training --train-split ${train_split} --val-split ${val_split} --mllm-mode ${mode} --use-cycle-loss

            # Combiner Evaluation
            python src/combiner_test.py --dataset ${dataset} --experiment-name ${exp_name} --projection-dim ${comb_proj} --hidden-dim ${comb_hidden} --clip-model-name ${clip_name} --clip-model-path ./models/${clip_name//\//-}/clip_finetuned_on_${dataset}_${exp_name}/saved_models/tuned_clip_$((clip_epochs - 1)).pt --transform ${transform} --val-split ${val_split} --combiner-model-path ./models/${clip_name//\//-}/combiner_trained_on_${dataset}_${exp_name}/saved_models/combiner_$((comb_epochs - 1)).pt --keep-selection ${keep_selection}
        done
    done
done