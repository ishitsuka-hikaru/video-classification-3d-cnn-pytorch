#!/bin/sh

#$ -l rt_G.large=1
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ~/venv/pytorch/bin/activate

# params
VIDEOS=~/repos/3D-ResNets-PyTorch/data_makehuman/makehuman_videos/mp4
MODEL=~/repos/3D-ResNets-PyTorch/data_makehuman/results/save_200.pth
ANNOTATIONS=~/repos/3D-ResNets-PyTorch/data_makehuman/makehuman.json
CLASSNAMES=~/repos/3D-ResNets-PyTorch/data_makehuman/class_names_list
RESULTS=~/repos/video-classification-3d-cnn-pytorch/results

# main
cd ..

# generate input video lists
echo -n > input
for f in $VIDEOS/*.mp4
do
    echo $f >> input
done

# generate class_names_list
echo -n > $CLASSNAMES
for l in `cat $ANNOTATIONS | jq -r .labels[]`
do
    echo $l >> $CLASSNAMES
done

# calculate class scores
NUM_CLASSES=`cat $CLASSNAMES | wc -l`
python3 main.py \
    --input input \
    --video_root $VIDEOS \
    --output output.json \
    --model $MODEL \
    --mode score \
    --n_classes $NUM_CLASSES

# visualize the classification results
cd generate_result_video
mkdir -p $RESULTS
python3 generate_result_video.py ../output.json $VIDEOS $RESULTS $CLASSNAMES 5

deactivate
