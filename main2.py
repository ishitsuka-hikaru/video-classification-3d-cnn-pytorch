import argparse
import json
import numpy as np
import os
import torch

from model import generate_model
from mean import get_mean
from classify import classify_video


def get_opts():
    p = argparse.ArgumentParser()
    p.add_argument('--video_path', type=str)
    p.add_argument('--annotation', type=str)
    p.add_argument('--model', type=str)
    p.add_argument('--output', type=str, default='output.json')
    p.add_argument('--mode', type=str, default='score', help='score or feature')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_threads', type=int, default=4)
    p.add_argument('--model_name', type=str, default='resnet')
    p.add_argument('--model_depth', type=int, default=34)
    p.add_argument('--resnet_shortcut', type=str, default='A')
    p.add_argument('--wide_resnet_k', type=int, default=2)
    p.add_argument('--resnext_cardinality', type=int, default=32)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--class_names_list', type=str, default=None)

    return p.parse_args()


if __name__ == '__main__':
    opt = get_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16

    with open(opt.annotation, 'r') as f:
        annotations = json.load(f)

    if opt.class_names_list:
        class_names = []
        with open(opt.class_names_list, 'r') as f:
            for l in f:
                class_names.append(l)
    else:
        class_names = annotations['labels']

    opt.n_classes = len(class_names)

    model = generate_model(opt)
    print(f'loading model {opt.model}')
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    
    video_paths = []
    for k, v in annotations['database'].items():
        if v['subset'] == 'testing':
            if 'video_path' in v.keys():
                video_path = v['video_path'].split('/')
                video_path = os.path.join(opt.video_path, *video_path[-3:])
            else:
                video_path = os.path.join(
                    opt.video_path, 'jpg',
                    v['annotations']['label'],
                    k
                )
            video_paths.append(video_path)
            
    outputs = []
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        if not os.path.exists(video_path):
            print(f'{video_name} not found')
            continue
        
        print(video_name)
        result = classify_video(video_path, video_name, class_names, model, opt)
        outputs.append(result)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
