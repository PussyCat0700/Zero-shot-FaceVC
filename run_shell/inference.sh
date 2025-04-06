#!/bin/bash

set -e

exp_name=zero_shot_facevc
inference_gpu=0

config_file=configs/zero_shot_facevc.ini
output_root=/data0/yfliu/outputs/zero_shot_facevc

python Tools/modify_config.py --config_file $config_file --inference_gpu $inference_gpu --output_root $output_root

python inference.py --config_file $config_file

parallel-wavegan-decode --checkpoint '/data0/yfliu/outputs/zero_shot_facevc/vqmivc/VQMIVC-pretrained models/vocoder/checkpoint-3000000steps.pkl' \
            --scp $output_root/test_samples_N+swapped/feats.1.scp --outdir $output_root/test_samples_N+swapped
parallel-wavegan-decode --checkpoint '/data0/yfliu/outputs/zero_shot_facevc/vqmivc/VQMIVC-pretrained models/vocoder/checkpoint-3000000steps.pkl' \
            --scp $output_root/test_samples_P+swapped/feats.1.scp --outdir $output_root/test_samples_P+swapped