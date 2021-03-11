#!/bin/sh

#$ -l rt_G.small=1
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ~/venv/pytorch/bin/activate

VIDEO_PATH=~/datasets/3D-ResNets-PyTorch/data_20210226/makehuman_videos
ANNOTATION=~/datasets/3D-ResNets-PyTorch/data_20210226/makehuman.json
# MODEL=../3D-ResNets-PyTorch/data_makehuman/results/save_200.pth
MODEL=pretrain_models/resnet-34-kinetics.pth
CLASS_NAMES_LIST=class_names_list

python main2.py \
    --video_path $VIDEO_PATH \
    --annotation $ANNOTATION \
    --model $MODEL \
    --class_names_list $CLASS_NAMES_LIST

deactivate
