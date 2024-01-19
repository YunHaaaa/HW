## Introduction


### Tab of Content
- [Installation](#1)
- [Usage](#2)
  - [Training](#3)
  - [Inference](#4)
- [Results](#5)

<span id="1"></span>

### Installation
1. Clone the repository
    ```sh
    git clone https://github.com/Harry-zzh/LGTM.git
    ```
2. Install the dependencies
    ```sh
    conda create -n lgtm python==3.8.0

    conda activate lgtm

    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

    cd LGTM

    pip install -r requirements.txt
    ```
<span id="2"></span>

### Usage

<span id="3"></span>

- #### Training
    ```sh
    python run_glue.py \
        --model_name_or_path google/bert_uncased_L-6_H-768_A-12 \
        --teacher_model bert-base-uncased \
        --task_name mnli \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --learning_rate 3e-05 \ # {3e-05,5e-05,1e-04}
        --t_learning_rate 3e-05 \
        --alpha_kd 1.0 \
        --temperature 1.0 \
        --num_train_epochs 6 \
        --output_dir <out_dir> \
        --eval_steps 1000 \
        --do_train \
        --do_eval \
        --train_teacher \
        --init_classifier_to_zero \
        --use_lgtm
    ```

    Hint: You can refer to some baselines using different settings:

    | Model | Setting                                              |
    | ----- | ---------------------------------------------------- |
    | KD    | (None)                                               |
    | Prokt | --init_classifier_as_zero --train_teacher            |
    | LGTM  | --init_classifier_as_zero --train_teacher --use_lgtm |

<span id="4"></span>

- #### Inference
    ```sh
    python run_glue.py \
        --model_name_or_path <checkpoint_path> \
        --task_name mnli \
        --per_device_eval_batch_size 32 \
        --output_dir <predict_out_dir> \
        --do_predict
    ```
    You can get the prediction files for each task and submit them into the GLUE test benchmark.

<span id="5"></span>

### Results

We give some results on validation set.

| Task          | Dev       |
| ------------- | --------- |
| MNLI (Acc.)   | 83.4      |
| QQP (F1/Acc.) | 91.1/88.0 |

### License
Distributed under the Apache 2.0 license. See LICENSE for more information.
