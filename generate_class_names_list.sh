#!/bin/sh
CLASSNAMES=class_names_list_makehuman
ANNOTATIONS=~/repos/3D-ResNets-PyTorch/data_makehuman/makehuman.json

echo -n > $CLASSNAMES
for l in `cat $ANNOTATIONS | jq -r .labels[]`
do
    echo $l >> $CLASSNAMES
done
