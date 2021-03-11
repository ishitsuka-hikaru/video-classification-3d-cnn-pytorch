#!/bin/sh

#$ -l rt_G.small=1
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ~/venv/pytorch/bin/activate

# params
INPUT=input
VIDEOS=videos
OUTPUT=output.json
MODEL=pretrain_models/resnet-34-kinetics.pth
CLASSNAMES=class_names_list
RESULTS=results

# generate input video lists
# echo -n > $INPUT
# for f in $VIDEOS/*.mp4
# do
#     echo `basename $f` >> $INPUT
#     # echo $f >> $INPUT
# done

# calculate class scores
python3 main.py \
    --input $INPUT \
    --video_root $VIDEOS \
    --output $OUTPUT \
    --model $MODEL \
    --mode score \
    --n_classes `cat $CLASSNAMES | wc -l`

# visualize the classification results
mkdir -p $RESULTS
python3 generate_result_video/generate_result_video.py $OUTPUT $VIDEOS $RESULTS $CLASSNAMES 5

deactivate
