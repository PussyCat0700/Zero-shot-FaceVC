#!/bin/bash

set -e

exp_name=zero_shot_facevc
inference_gpu=0

config_file=configs/zero_shot_facevc.ini
output_root=./output/zero_shot_facevc

python Tools/modify_config.py --config_file $config_file --inference_gpu $inference_gpu --output_root $output_root


python inference.py --config_file $config_file || exit 