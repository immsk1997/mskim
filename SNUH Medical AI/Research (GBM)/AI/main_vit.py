"""code Ref"""
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# https://github.com/kyuchoi/3D_MRI_survival_glioma/tree/main/model/utils.py

"""Paper & Ref"""
# https://arxiv.org/abs/2010.11929 -> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
# https://kmhana.tistory.com/27
# https://m.blog.naver.com/nueyet/222984347342

#%%
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import glob
import random
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel as nib
import pandas as pd
import argparse

import math
import torch # For building the networks 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as summary
import torchvision.transforms as transforms

import monai
from monai.networks.nets import *
from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import argparse

from medcam import medcam
from medcam import *

import matplotlib
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize
from scipy import ndimage

from vit_3d import *
from utils_vit import *

''' Setting '''
#%%
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123456) # 12347541
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') 
    parser.add_argument('--ext_dataset_name', type=str, default='SNUH')
    parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets')
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    return parser

main_args = get_args_parser().parse_args()
args = config()

breaks, n_intervals = get_n_intervals(fixed_interval_width = False) # True #
args.n_intervals = n_intervals

now = datetime.now()
exp_time = now.strftime("%Y_%m_%d_%H_%M")

DL_score_dir = os.path.join(args.exp_dir, 'DL_features', f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(DL_score_dir, exist_ok = True)
args.DL_score_dir = DL_score_dir

attention_map_dir = os.path.join(args.exp_dir, 'attention_maps', f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(attention_map_dir, exist_ok = True)
args.attention_map_dir = attention_map_dir

exp_path = os.path.join(args.exp_dir, f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(exp_path, exist_ok = True)
print_args(main_args, exp_path)

gpu_id = main_args.gpu_id
print(f'Training on GPU {gpu_id}')

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

args.dataset_name = '_'.join(main_args.dataset_list)
print(f'Train dataset_name:{args.dataset_name}')

test_gpu_id = main_args.test_gpu_id # int(main_args.gpu_id + 1) # 
print(f'Testing on GPU {test_gpu_id}')

test_device = torch.device(test_gpu_id)

to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
torch.cuda.empty_cache()

os.environ['MKL_THREADING_LAYER'] = 'GNU' # in Linux, I had to write a script to call "export MKL_THREADING_LAYER=GNU" (which sets that environment variable)
set_seed(main_args.seed)
print(f'Setting seed:{main_args.seed}')

get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv') #  'manual_labels', #
get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))

# combine_img : 여러 디렉토리에 있는 nifti 파일들 한군데로 뭉치기
combine_img(main_args, args)

def make_kfold_df_proc_labels(args, dataset_name, remove_idh_mut = False, fixed_interval_width = 0):
  
  df = pd.read_csv(get_label_path(dataset_name), dtype='string')
  df = df.set_index('ID')
  df = df.sort_index(ascending=True)
  df = df[args.data_key_list]
  print(f'df.index.values:{len(df.index.values)}')
  
  if remove_idh_mut:
    condition = df.IDH.astype(int) == 0 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'after removing idh mutation; df.index.values:{len(df.index.values)}')
  
  if '_' in dataset_name:
    list_dataset = dataset_name.split('_')
    print(f'list_dataset: {list_dataset}')
    
    comm_list = []
    for split_dataset in list_dataset:
      print(f'split_dataset: {split_dataset}')
      img_dir = os.path.join(args.data_dir, split_dataset, "VIT",f'{args.compart_name}_BraTS')
      split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
      print(f'split_img_label_comm_list:{len(split_comm_list)}')
      comm_list.extend(split_comm_list)
    
  else:  
    img_dir = os.path.join(args.data_dir, dataset_name, "VIT",f'{args.compart_name}_BraTS') 
    comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
  
  print(f'img_label_comm_list:{len(comm_list)}')
  print(f'dataset_name:{dataset_name}, {len(comm_list)}') # SNUH_UPenn, 1113
  
  df = df.loc[sorted(comm_list)] #.astype(int) 

  print(f'{dataset_name} df.shape: {df.shape}') # (1113, 8) 

  ID, duration, event = get_target(df)
  
  kfold = add_kfold_to_df(df, args, main_args.seed)
  
  breaks, _ = get_n_intervals(fixed_interval_width)

  proc_labels = make_surv_array(duration, event, breaks)
  df_proc_labels = pd.DataFrame(proc_labels)

  df_proc_labels['ID'] = ID
  df_proc_labels['kfold'] = kfold
  df_proc_labels = df_proc_labels.set_index('ID')
  
  proc_label_path = os.path.join(args.proc_label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
  df_proc_labels.to_csv(proc_label_path)
  
  return df_proc_labels, event, duration
#%%
df_proc_labels_train, event_train, duration_train = make_kfold_df_proc_labels(args, f'{args.dataset_name}')
df_proc_labels_test, event_test, duration_test = make_kfold_df_proc_labels(args, f'{main_args.ext_dataset_name}')

# %%
'''모델 및 학습 방법 선정'''
model = vit_large_patch16(args=args,img_size=120,patch_size=6,in_chans=4,num_classes=args.n_intervals).to(device)
model_architect = vars(model)["_modules"]
print(f'model_architect:{model_architect}')

layer_name = [n for n in model_architect]
print(f'layer_name:{layer_name}')

base_optimizer = AdamP
optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay) # SAM 역할 : 최적화 과정 안정화 (보조적인 역할)
criterion = nnet_loss # TaylorCrossEntropyLoss(n=2,smoothing=0.2)
scheduler = fetch_scheduler(optimizer)

'''train, valid data'''
valid_df = df_proc_labels_train[df_proc_labels_train.kfold == 0]
train_df = df_proc_labels_train[df_proc_labels_train.kfold != 0]

train_df_path = os.path.join(args.proc_label_dir, f'train_df_proc_labels_{args.dataset_name}.csv')
train_df.to_csv(train_df_path)

valid_df_path = os.path.join(args.proc_label_dir, f'valid_df_proc_labels_{args.dataset_name}.csv')
valid_df.to_csv(valid_df_path)
    
train_data = CustomDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}')
valid_data = CustomDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}')
  
dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
}
    
print(f'num of train_data: {len(train_data)}')
print(f'num of valid_data: {len(valid_data)}')
    
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
}

model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs=10, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device, fold=0)

'''Test Data & Test Model'''
model = vit_large_patch16(args=args,img_size=120,patch_size=6,in_chans=4,num_classes=args.n_intervals).to(test_device)
model = load_ckpt(args, model)

proc_label_path_test = os.path.join(args.proc_label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
df_proc_labels_test = pd.read_csv(proc_label_path_test, dtype='string')
df_proc_labels_test = df_proc_labels_test.set_index('ID')

external_data = CustomDataset(df = df_proc_labels_test, args = args, dataset_name=f'{main_args.ext_dataset_name}')
test_loader = DataLoader(dataset=external_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

'''Survival Analysis'''  
df_DL_score_test = df_proc_labels_test.copy()
df_DL_score_test.drop(columns=df_DL_score_test.columns,inplace=True) # index 열 제외하고 모두 삭제

for i in np.arange(n_intervals):
  df_DL_score_test.insert(int(i), f'MRI{i+1}', '')
df_DL_score_test.insert(n_intervals, 'oneyr_survs_test', '')

oneyr_survs_test = []
for subj_num, (inputs,labels) in enumerate(test_loader):
  model.eval()
  inputs = inputs.to(test_device)
  labels = labels.to(test_device)

  y_pred = model(inputs) # torch.Size([4, 19])
  print(f'y_pred:{y_pred}')
  print(f'labels:{labels}')
  print(f'subj_num:{subj_num}')
  
  '''C-index, Brier Score'''
  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
  # breaks=np.arange(0.,365.*5,365./8)
  y_pred_np = to_np(y_pred)

  cumprod = np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)
  oneyr_surv_test = cumprod[:,-1]
  print(f'oneyr_surv_test: {oneyr_surv_test}')
  
  DL_scores = []
  for n_interval in np.arange(1, n_intervals+1):
    DL_score = np.cumprod(y_pred_np[:,0:n_interval], axis=1)[:,-1][0]
    # print(f'DL_score_{n_interval}th_term_oneyr_surv_test:{DL_score}')  
    DL_scores.append(DL_score)
  DL_scores.append(oneyr_surv_test[0])
  # print(f'DL_scores:{len(DL_scores)}') # 19+1=20
  df_DL_score_test.loc[df_DL_score_test.index[subj_num]] = DL_scores
  oneyr_survs_test.extend(oneyr_surv_test)

print(f'df_DL_score_test.shape:{df_DL_score_test.shape}')
DL_score_path = os.path.join(DL_score_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_DL_score_s{main_args.seed}.csv')
df_DL_score_test.to_csv(DL_score_path)

oneyr_survs_test = np.array(oneyr_survs_test)
#%%
print(f'duration_test.shape:{duration_test.shape}')
print(f'oneyr_survs_test.shape:{oneyr_survs_test.shape}')
print(f'event_test.shape:{event_test.shape}')

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_test, oneyr_survs_test, event_test)

print(f'Original C-index for valid: {original_c_index:.4f}')
print(f'95% CI for C-index for valid: ({ci_lower:.4f}, {ci_upper:.4f})')

score_test = get_BS(event_test, duration_test, oneyr_survs_test)