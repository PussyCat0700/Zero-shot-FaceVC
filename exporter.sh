#!/bin/bash

#SBATCH --job-name=exporter
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=20  # 每个进程的CPU数量
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=all     # 设置邮件通知类型，可选all, end, fail, begin
#SBATCH --mail-user=1729372667@qq.com # 设置通知邮箱

set -e

# python Tools/preprocess/extract_wav_feature.py
# python Tools/preprocess/extract_face_feature.py
bash ./run_shell/inference.sh