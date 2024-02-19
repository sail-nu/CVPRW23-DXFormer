import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--arch', default='dda_ca_enc')
parser.add_argument('--version', default='exp4')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--epoch', default='50')
parser.add_argument('--feature', default='I3D')

args = parser.parse_args()
 
num_epochs = int(args.epoch)
    
lr = 0.0005
num_layers = 10
num_f_maps = 64
bz = 1

if args.feature == 'pose':
    features_dim = 126
elif args.feature == 'clip':
    features_dim = 512
elif args.feature == 'bridge_pose':
    features_dim = 894   
elif args.feature.split('_')[0] == 'bridge' or args.feature.split('_')[0] == 'ViT':
    features_dim = 768
else:
    features_dim = 2048

channel_mask_rate = 0.3
arch_type = args.arch
pos_enc = None #"learnable" #"fixed"  

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
    lr = 0.001
    bz = 4

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    sample_rate = 4
    lr = 0.001
    bz = 12

if args.dataset == 'hoi4d':
    lr = 0.001
    bz = 8

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"+args.feature+"/"
if args.feature.split('_')[0] == 'bridge': 
    features_path = "./data/"+args.dataset+"/features/"+args.feature+"_"+args.split+"/"
gt_path = "./data/"+args.dataset+"/groundTruth/"
 
mapping_file = "./data/"+args.dataset+"/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split + '/' + args.version

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split+ '/' + args.version

 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, arch_type, pos_enc)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

