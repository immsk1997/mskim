import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary as summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchio as tio

import os
import glob
import copy
from datetime import datetime
import cv2
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import json
import argparse
import nibabel as nib

from monai.transforms import RandFlip, Rand3DElastic, RandAffine, RandGaussianNoise, AdjustContrast, RandSpatialCrop # Rand3DElastic
from sklearn.model_selection import train_test_split, StratifiedKFold
from distutils.dir_util import copy_tree
from tqdm import tqdm
from PIL import Image

import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import plotly.graph_objects as go
from IPython.display import HTML

from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from skimage.transform import resize

#%%

''' lambda '''
to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

# get_dataset_name = lambda dataset_list: '_'.join(dataset_list)
get_dir = lambda directory: [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
convert_list2df = lambda idlist: pd.DataFrame(idlist, columns = ['ID'], dtype='string').set_index('ID')

#%%

def print_args(args, exp_path):
  args_path = os.path.join(exp_path,'commandline_args.txt')
  with open(args_path, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

  with open(args_path, 'r') as f:
      args.__dict__ = json.load(f)

class config(object):
  
  def __init__(self):
    self.scheduler = 'CosineAnnealingLR'
    self.T_max = 10
    self.T_0 = 10
    self.lr = 1e-4 # ORIGINAL: 1e-4
    self.min_lr = 1e-6
    self.weight_decay = 1e-5
    self.n_fold = 10 # 5
    self.smoothing = 0.3
    self.batch_size = 16 # CNNs : 64, Transformers : 16
    
    self.net_architect = 'VisionTransformer' # 'DenseNet' # 'resnet50_cbam' #'SEResNext50' # VisionTransformer
    self.finetuning_type = 'classifier' # None # 'classifier'
    self.compart_name = 'resized' # 'seg' # 
    self.sequence = ['t1','t2','t1ce','flair']
    # self.spec_event = 'death' # 'prog' # 
    
    self.data_key_list = ['sex', 'age', 'IDH', 'glioma_type','glioma_num','MGMT', 'GBL', 'EOR', 'duration_death', 'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion'] # 'KPS': ONLY in UPenn, severance, NOT in SNUH, TCGA; 'prog': NOT in UPenn, TCGA, ONLY in SNUH, severance, TCGA; 'EOR': NOT in TCGA
         
    self.root_dir = r'/mnt/hdd3/mskim/GBL'
    self.data_dir = os.path.join(self.root_dir, 'data')
    self.label_dir = os.path.join(self.data_dir, 'label', 'surv_labels')
    self.proc_label_dir = os.path.join(self.label_dir, 'proc_labels')
    os.makedirs(self.proc_label_dir, exist_ok = True)
    
    self.exp_dir = os.path.join(self.root_dir, 'code', 'experiment')
    os.makedirs(self.exp_dir, exist_ok = True)

    self.dataset_name = ''
    
args = config()
print(f'args:{args.__dict__}')

print(f'Using {args.net_architect}')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# set_seed(args.seed)

#%%
# min-max scaling of img and convert to uint8 into range of 0-255
def min_max_norm(img):
  
  img = img.astype(np.float32)
  img = (img-img.min())/(img.max()-img.min())
  img = (img*255).astype(np.uint8)
  img = np.stack((img,)*3, axis=-1)
  
  return img

def superimpose_img(img, heatmap, alpha = 0.3):
  
  grad_heatmap = resize(heatmap, (img.shape[0], img.shape[1], img.shape[2]))

  cmap = plt.cm.jet
  grad_heatmap_rgb = cmap(grad_heatmap)
  grad_heatmap_rgb = grad_heatmap_rgb[...,:3]
  grad_heatmap_rgb = np.uint8(grad_heatmap_rgb * 255)

  grad_result = grad_heatmap_rgb * alpha + img * (1 - alpha) #.astype(np.uint8)
  grad_result = grad_result / np.max(grad_result)

  # print(f'range of grad_result: {np.min(grad_result)}-{np.max(grad_result)}') # 0.0 1.0
  # print(f'shape of grad_result: {grad_result.shape}')
  return grad_result, grad_heatmap

def plot_slices_superimposed(data, x_slice, y_slice, z_slice, use_midline = True):
    
    matplotlib.rcParams['animation.embed_limit'] = 500
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    if use_midline:
      xslice = data.shape[0] // 2 # specify
      yslice = data.shape[1] // 2 # specify
      zslice = data.shape[2] // 2 # specify
    else:
      xslice = x_slice
      yslice = y_slice
      zslice = z_slice

    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Plot slices with alpha = 0.5 for some transparency
    scatter = ax.scatter(x[mask], y[mask], z[mask], c=data[mask], s=20, cmap = 'gray') # norm=norm, # , cmap = 'jet'

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Invert axis
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.invert_zaxis()
    
    # make 2 rotations with different directions
    total_frames = 720  # for two rounds # 360 #

    def update(num):
        
        # Angles for first rotation
        final_azim_1 = 300
        final_elev_1 = 285

        # Angles for second rotation
        final_azim_2 = 600
        final_elev_2 = 570

        if num < total_frames / 2:
            azim = (final_azim_1 / (total_frames / 2)) * num
            elev = (final_elev_1 / (total_frames / 2)) * num
        else:
            azim = final_azim_1 + ((final_azim_2 - final_azim_1) / (total_frames / 2)) * (num - total_frames / 2)
            elev = final_elev_1 + ((final_elev_2 - final_elev_1) / (total_frames / 2)) * (num - total_frames / 2)

        ax.view_init(elev=elev, azim=azim)
        return scatter
    
   
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, total_frames, 1), interval=60)
    html = ani.to_jshtml() # save as html 
    
    # return ani # save as gif
    return html#, ani

def plotly_slices_superimposed(data):
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    # Slice indices
    xslice = data.shape[0] // 2 # specify
    yslice = data.shape[1] // 2 # specify
    zslice = data.shape[2] // 2 # specify
    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Ensure there's data to plot
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x[mask].flatten(),
        y=y[mask].flatten(),
        z=z[mask].flatten(),
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color=data[mask].flatten()
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

#%%

def get_n_intervals(fixed_interval_width = False):

  if fixed_interval_width:
    breaks=np.arange(0.,365.*5,365./8)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    # print(f'n_intervals: {n_intervals}') # 19
  else:
    halflife=365.*2
    breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    # print(f'n_intervals: {n_intervals}') # 19

    return breaks, n_intervals

# get 95% confidence interval of concordance index using bootstrap
def bootstrap_cindex(time, prediction, event, n_iterations=1000):
    # Compute the original C-index
    original_c_index = concordance_index(time, prediction, event)

    # Initialize a list to store bootstrapped C-indexes
    bootstrap_c_indexes = []

    # Perform bootstrapping
    for i in range(n_iterations):
        # Resample with replacement
        resample_indices = resample(np.arange(len(time)), replace=True)
        time_sample = time[resample_indices]
        event_sample = event[resample_indices]
        prediction_sample = prediction[resample_indices]

        # Compute the C-index on the bootstrap sample
        c_index_sample = concordance_index(time_sample, prediction_sample, event_sample)

        bootstrap_c_indexes.append(c_index_sample)

    # Compute the 95% confidence interval for the C-index
    ci_lower = np.percentile(bootstrap_c_indexes, 2.5)
    ci_upper = np.percentile(bootstrap_c_indexes, 97.5)

    return original_c_index, ci_lower, ci_upper

#%%

''' copying and combining images of dataset_list '''

def combine_img(main_args, args):
  dataset_name = '_'.join(main_args.dataset_list)
  
  if args.net_architect =='VisionTransformer':
    target_dataset_path = os.path.join(args.data_dir, dataset_name, "VIT",f'{args.compart_name}_BraTS')
    os.makedirs(target_dataset_path, exist_ok=True)
  
  elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
    target_dataset_path = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS')
    os.makedirs(target_dataset_path, exist_ok=True)
  
  if len(os.listdir(target_dataset_path)) != 0:
    print(f"Already copyied images of {dataset_name} for training to {target_dataset_path} path")
  
  else:
    for dataset in main_args.dataset_list:
      print(f"copying images of {dataset} for training to {target_dataset_path} path")
      
      if args.net_architect == 'VisionTransformer':
        img_dataset_path = os.path.join(args.data_dir, dataset, "VIT",f'{args.compart_name}_BraTS')
      elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
        img_dataset_path = os.path.join(args.data_dir, dataset, f'{args.compart_name}_BraTS')
      
      for img_dir in tqdm(os.listdir(img_dataset_path)):
        img_dir_path = os.path.join(img_dataset_path, img_dir)
        print(f'img_dir_path:{img_dir_path}')
        os.makedirs(os.path.join(target_dataset_path, img_dir), exist_ok=True)
        copy_tree(img_dir_path, os.path.join(target_dataset_path, img_dir))

#%%

''' getting together multiple (i.e. SNUH, severance, UPenn) ${dataset}_OS_all.csv files into final csv indexing only 1) GBL vs all; and 2) 1yr vs OS, and save them into anoter .csv file '''

def save_label_dataset_list(main_args, args):
  
  df = pd.DataFrame()
  
  for dataset in main_args.dataset_list:
    print(f'dataset:{dataset} for training')
    
    if args.net_architect == 'VisionTransformer':
      df_dataset_path = os.path.join(args.label_dir, f'{dataset}_OS_all_vit.csv')
    
    elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
      df_dataset_path = os.path.join(args.label_dir, f'{dataset}_OS_all.csv')
    
    df_data = pd.read_csv(df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
    df_data = df_data.set_index('ID')
    df_data = df_data.sort_index(ascending=True)
        
    df_dataset = df_data[args.data_key_list]
    print(f'df_dataset.shape:{df_dataset.shape}')
    # df_label_dataset_list = pd.merge(df_dataset, df_label_dataset_list, left_on='ID', right_index=True) # NOT WORKING
    df = pd.concat([df_dataset, df])
  print(f'df_label_dataset_list.shape:{df.shape}') # 
  # print(f'df.head:{df.head(10)}')

  dataset_name = '_'.join(main_args.dataset_list)
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(df.index.values)} cases')
    condition = df.GBL.astype(int) == 1 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(df.index.values)} cases')
    print(f'df.columns:{df.columns}')
    if "biopsy_exclusion" in df.columns:
      condition = df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = df[condition].index.tolist() 
      df = df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        df = df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(df['event_death'].sum())
        df.loc[(df['event_death'] == 1) & (df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(df['event_death'].sum())
    
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        df = df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(df['event_prog'].sum())
        df.loc[(df['event_prog'] == 1) & (df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(df['event_prog'].sum())
        
    else:
        pass

  df_path = os.path.join(args.label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  
  if args.net_architect =='VisionTransformer':
    df_path = os.path.join(args.label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  
  df.to_csv(df_path)
  
  print(f'saving new label csv file for {dataset_name} at {df_path}') # 

  return df

def save_label_ext_dataset(main_args, args):
  
  ext_df = pd.DataFrame()
  
  print(f'dataset:{main_args.ext_dataset_name} for training')
  
  if args.net_architect =='VisionTransformer':
    ext_df_dataset_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_OS_all_vit.csv')
  elif args.net_architect =='DenseNet' or 'resnet50_cbam' or 'SEResNext50':
    ext_df_dataset_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_OS_all.csv')
  
  ext_df_data = pd.read_csv(ext_df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
  ext_df_data = ext_df_data.set_index('ID')
  ext_df_data = ext_df_data.sort_index(ascending=True)
    
  ext_df = ext_df_data[args.data_key_list]
  print(f'ext_df_dataset.shape:{ext_df.shape}')
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(ext_df.index.values)} cases')
    condition = ext_df.GBL.astype(int) == 1
    filtered_ID = ext_df[condition].index.tolist() 
    ext_df = ext_df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(ext_df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(ext_df.index.values)} cases')
    if "biopsy_exclusion" in ext_df.columns:
      condition = ext_df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = ext_df[condition].index.tolist() 
      ext_df = ext_df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(ext_df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(ext_df['event_death'].sum())
        ext_df.loc[(ext_df['event_death'] == 1) & (ext_df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_death'].sum())
        
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(ext_df['event_prog'].sum())
        ext_df.loc[(ext_df['event_prog'] == 1) & (ext_df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_prog'].sum())
        
    else:
        pass

  if args.net_architect =='VisionTransformer':
    ext_df_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
    ext_df.to_csv(ext_df_path)
  
  elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
    ext_df_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
    ext_df.to_csv(ext_df_path)
  
  print(f'saving new label csv file for external set {main_args.ext_dataset_name} at {ext_df_path}') # 

  return ext_df

#%%
def make_kfold_df_proc_labels(main_args, args, dataset_name, remove_idh_mut = False, fixed_interval_width = 0):
  
  if args.net_architect =='VisionTransformer':
    get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  elif args.net_architect =='DenseNet' or 'resnet50_cbam' or 'SEResNext50':  
    get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  
  get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))
  
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
      
      if args.net_architect =='VisionTransformer':
         img_dir = os.path.join(args.data_dir, split_dataset, "VIT",f'{args.compart_name}_BraTS')
         split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
         print(f'split_img_label_comm_list:{len(split_comm_list)}')
         comm_list.extend(split_comm_list)
      
      elif args.net_architect =='DenseNet' or 'SEResNext50' or 'resnet50-cbam':
        img_dir = os.path.join(args.data_dir, split_dataset, f'{args.compart_name}_BraTS')
        split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
        print(f'split_img_label_comm_list:{len(split_comm_list)}')
        comm_list.extend(split_comm_list)
      
  else:  
    if args.net_architect =="VisionTransformer":
       img_dir = os.path.join(args.data_dir, dataset_name, "VIT",f'{args.compart_name}_BraTS')
       comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
       
    elif args.net_architect =='DenseNet' or 'SEResNext50' or 'resnet50-cbam':
       img_dir = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS') 
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
def make_class_df(main_args,args, dataset_name, remove_idh_mut = False):
  
  get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                           np.array(df["glioma_num"].tolist(), dtype=int))
  
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
      if args.net_architect =='VisionTransformer':
         img_dir = os.path.join(args.data_dir, split_dataset, "VIT",f'{args.compart_name}_BraTS')
      else:
         img_dir = os.path.join(args.data_dir, split_dataset, f'{args.compart_name}_BraTS')
      
      split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
      print(f'split_img_label_comm_list:{len(split_comm_list)}')
      comm_list.extend(split_comm_list)
    
  else:  
    img_dir = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS') 
    comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
  
  print(f'img_label_comm_list:{len(comm_list)}')
  print(f'dataset_name:{dataset_name}, {len(comm_list)}') # SNUH_UPenn, 1113
  
  df = df.loc[sorted(comm_list)] #.astype(int) 

  print(f'{dataset_name} df.shape: {df.shape}') # (1113, 8) 

  ID, glioma_class = get_target(df)
  
  copy_df = df.copy()
  copy_df["ID"] = ID
  copy_df["glioma_num"] = glioma_class
  
  class_df = pd.concat([copy_df["ID"],copy_df["glioma_num"]],axis=1)
  class_df = class_df.set_index("ID")
  class_label_path = '/mnt/hdd3/mskim/GBL/data/label/class_labels'
  class_labels_path = os.path.join(class_label_path,f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_class_labels.csv')
  class_df.to_csv(class_labels_path)
  
  return class_df
  
#%%
def nnet_pred_surv(y_pred, breaks, fu_time):

  y_pred=np.cumprod(y_pred, axis=1)
  pred_surv = []
  for i in range(y_pred.shape[0]):
    pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))
  return np.array(pred_surv)

def add_kfold_to_df(df, args, seed):
  
  ''' Create Folds 
  ref:
  1) https://www.kaggle.com/code/debarshichanda/seresnext50-but-with-attention 
  2) https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
  '''  
  
  skf = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=seed)
  for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.event_death)):
      # print(fold, val_)
      # print(df.index[val_])
      df.loc[df.index[val_] , "kfold"] = int(fold)
      
  df['kfold'] = df['kfold'].astype(int)
  kfold = df['kfold'].values
  
  return kfold

def random_split(id_list, split_ratio):
  ''' df: dataframe for total dataset '''
  n_sample = len(id_list) 
  id_list = sorted(id_list)
  train_nums = np.random.choice(n_sample, size = int(split_ratio * n_sample), replace = False)
  print(f'train_nums:{len(train_nums)}')
  val_nums = [num for num in np.arange(n_sample) if num not in train_nums]
  
  return train_nums, val_nums

def make_surv_array(t,f,breaks):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train

#%%

def fetch_scheduler(optimizer):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr)
    elif args.scheduler == None:
        return None
        
    return scheduler

#%%

def run_fold(df, args, model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]
    
    '''
    df_proc_labels_test 를 그냥 .csv 로 저장하고 load 하는 방식으로 하기
    '''
    train_df_path = os.path.join(args.proc_label_dir, f'train_df_proc_labels_{args.dataset_name}.csv')
    train_df.to_csv(train_df_path)

    valid_df_path = os.path.join(args.proc_label_dir, f'valid_df_proc_labels_{args.dataset_name}.csv')
    valid_df.to_csv(valid_df_path)
    
    train_data = SurvDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.train_transform, aug_transform=True) #True)
    valid_data = SurvDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.valid_transform, aug_transform=False)
  
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

    model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
    return model, history

#%%
def run_fold_vit(df, args, model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]

    train_data = ViTDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}')
    valid_data = ViTDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}')
  
    dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
    }
    
    print(f'num of train_data: {len(train_data)}')
    print(f'num of valid_data: {len(valid_data)}')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
    return model, history

#%%
def train_classification(df, args, model, criterion, optimizer, scheduler, device, num_epochs=10):
    train_df = df

    train_data = ClassViTDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}')
    
    dataset_sizes = {
        'train' : len(train_data)
      }
    
    print(f'num of train_data: {len(train_data)}')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    
    dataloaders = {
        'train' : train_loader
      }

    model, history = train_classification_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device)
    
    return model, history
#%%

def get_transform(args, dataset_name):
  
  if dataset_name == 'SNUH_UPenn_TCGA':
    landmark_dataset_name = 'SNUH_UPenn_TCGA' # train/valid/test=0.75/0.72/0.69 # 
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  else:
    landmark_dataset_name = dataset_name
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  
  landmarks_dir = os.path.join(args.data_dir, 'histograms', landmark_dataset_name)
  
  landmarks = {}
  for seq in args.sequence:
    
    seq_landmarks_path = os.path.join(landmarks_dir, f'{seq}_histgram.npy')
    landmarks[f'{seq}'] = seq_landmarks_path
  
  # print(f'landmarks:{list(landmarks.keys())}') # ['t1', 't2', 't1ce', 'flair']
    
  basic_transforms = [
      tio.HistogramStandardization(landmarks), 
      # tio.ZNormalization() # (masking_method=lambda x: x > 0) # x.mean() # # NOT working: RuntimeError: Standard deviation is 0 for masked values in image    
  ]

  basic_transform = tio.Compose(basic_transforms)
  # aug_transform = Compose(aug_transforms)
  
  print(f'transform for {dataset_name} was obtained')
  
  return basic_transform

#%%
def get_transform_vit(args, dataset_name):
  
  if dataset_name == 'SNUH_UPenn_TCGA':
    landmark_dataset_name = 'SNUH_UPenn_TCGA' # train/valid/test=0.75/0.72/0.69 # 
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  else:
    landmark_dataset_name = dataset_name
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  
  landmarks_dir = os.path.join(args.data_dir, 'histograms', landmark_dataset_name,"VIT")
  
  landmarks = {}
  for seq in args.sequence:
    
    seq_landmarks_path = os.path.join(landmarks_dir, f'{seq}_histgram.npy')
    landmarks[f'{seq}'] = seq_landmarks_path
  
  # print(f'landmarks:{list(landmarks.keys())}') # ['t1', 't2', 't1ce', 'flair']
    
  basic_transforms = [
      tio.HistogramStandardization(landmarks), 
      # tio.ZNormalization() # (masking_method=lambda x: x > 0) # x.mean() # # NOT working: RuntimeError: Standard deviation is 0 for masked values in image    
  ]

  basic_transform = tio.Compose(basic_transforms)
  # aug_transform = Compose(aug_transforms)
  
  print(f'transform for {dataset_name} was obtained')
  
  return basic_transform

#%%
def load_ckpt(args, model):
  ckpt_dir = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
  os.makedirs(ckpt_dir, exist_ok = True)
  ckpt_list = glob.glob(f'{ckpt_dir}/*.pth') 
  ckpt_model = max(ckpt_list, key=os.path.getctime)
  print(f'latest_ckpt_model: {ckpt_model}') #'Fold0_3.1244475595619647_epoch23.pth'
  ckpt_path = os.path.join(ckpt_dir, ckpt_model) 
  model_dict = torch.load(ckpt_path, map_location='cuda') # NOT working when in utils.py: f'cuda:{gpu_id}'
  model.load_state_dict(model_dict)
  return model

class SurvDataset(nn.Module):
  def __init__(self, df, args, dataset_name, transforms=None, aug_transform=False): # ['27179925', '45163562', 'UPENN-GBM-00291_11', '42488471', 'UPENN-GBM-00410_11', '28802482']
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    # print(self.df.shape) # (890, 39) # (223, 39)
    self.img_dir = os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_BraTS') # 'SNUH_UPenn_TCGA_severance'
    self.transforms = transforms
    self.aug_transform = aug_transform

    self.znorm = tio.ZNormalization()
    self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    self.crop_size = 64
    self.crop = RandSpatialCrop(roi_size=(self.crop_size, self.crop_size, self.crop_size), random_size=False)
    
    # self.rand_affiner = RandAffine(prob=0.9, rotate_range=[-0.5,0.5], translate_range=[-7,7],scale_range= [-0.15,0.1], padding_mode='zeros')
    self.rand_affiner = RandAffine(prob=0.9)
    self.rand_elastic = Rand3DElastic(prob=0.8, magnitude_range = [-1,1], sigma_range = [0,1])
    self.flipper1 = RandFlip(prob=0.5, spatial_axis=0)
    self.flipper2 = RandFlip(prob=0.5, spatial_axis=1)
    self.flipper3 = RandFlip(prob=0.5, spatial_axis=2)
    self.gaussian = RandGaussianNoise(prob=0.3)
    self.contrast = AdjustContrast(gamma=2)
    self.compart_name = args.compart_name
  
  def concat_seq_img(self, x):
      return torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], axis=0)
  
  def __len__(self):
    return len(self.df) # 결국 df 로 index 를 하기 때문에 dataset의 길이도 len(df): df를 train_df, val_df 넣는것에 따라 dataset이 train_set, val_set이 됨.

  def augment(self, img):
    img = self.crop(img)
    img = self.gaussian(img)
    
    return img

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
    kfold = self.df['kfold'][idx]
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')), # t1_seg.nii.gz
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')), 
            
        )   
    
    if self.transforms:
      subject = self.transforms(subject)
    
    img = self.concat_seq_img(subject)
    # print(f'img loaded: {img.shape}') # torch.Size([4, 12])
    
    if self.aug_transform:
      img = self.augment(img)
    # print(f'final input image shape: {img.shape}') # torch.Size([4, 120, 120, 78])
    
    proc_label_list = list(self.df[self.df.columns.difference(['ID', 'kfold'])].iloc[idx].values) # ID, kfold 는 제외
    proc_labels = [int(float(proc_label)) for proc_label in proc_label_list] # '1.0' -> 1: string -> float -> int
    proc_labels = torch.tensor(proc_labels)
    # print(f'proc_labels.shape:{proc_labels.shape}') # torch.Size([38])
    
    return img, proc_labels

class ViTDataset(nn.Module):
  def __init__(self, df, args, dataset_name):
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    self.img_dir = os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS') # 'SNUH_UPenn_TCGA_severance'
    self.compart_name = args.compart_name
    
  def concat_seq_img(self, x):
      seq_cat = torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], dim=0)
      return seq_cat.to(torch.float32)
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
    kfold = self.df['kfold'][idx]
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')),
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')),
        )

    img = self.concat_seq_img(subject)
    # print(f'img loaded: {img.shape}') -> torch(4,224,224,224)
    
    proc_label_list = list(self.df[self.df.columns.difference(['ID', 'kfold'])].iloc[idx].values) # ID, kfold 는 제외
    proc_labels = [int(float(proc_label)) for proc_label in proc_label_list] # '1.0' -> 1: string -> float -> int
    proc_labels = torch.tensor(proc_labels)
    # print(f'proc_labels.shape:{proc_labels.shape}') torch(38)
    return img, proc_labels

class ClassViTDataset(nn.Module):
  def __init__(self, df, args, dataset_name):
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    self.img_dir = os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS') # 'SNUH_UPenn_TCGA_severance'
    self.compart_name = args.compart_name
    
  def concat_seq_img(self, x):
      seq_cat = torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], dim=0)
      return seq_cat.to(torch.float32)
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')),
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')),
        )

    img = self.concat_seq_img(subject)
    
    glioma_class_num = self.df["glioma_num"].iloc[idx]
    class_label = int(glioma_class_num) # wild : 1 , astrocytoma : 2 , oligodendro : 3, mutant : 4, gliosarcoma : 5 (only severance)
    class_label = torch.tensor(class_label-1, dtype=torch.float32) # wild : 0 , astrocytoma : 1 , oligodendro : 2, mutant : 3, gliosarcoma : 4 (only severance)
    
    return img, class_label

#%%
class PropHazards(nn.Module):
  def __init__(self, size_in, size_out):#, device):
    super().__init__()
    self.linear = nn.Linear(size_in, size_out)#.to(device)

  def forward(self, x):
    x = self.linear(x)
    
    return torch.pow(torch.sigmoid(x), torch.exp(x)) #.float().to(device)

class CustomNetwork(nn.Module):
  def __init__(self, args, base_model):
    super().__init__()
    self.base_model = base_model
    self.model = vars(base_model)['_modules'] #["_modules"] = nn.Module 상속받고 모델 구조만 확인하고 싶을 때, vars = Dictionary
    
    self.output_dim = args.n_intervals # 19
    print(f'self.output_dim:{self.output_dim}')
    layers = []
    self.layer_name_list = [n for n in self.model][:-1]
    print(f'self.layer_name_list:{self.layer_name_list}')

    for name in self.layer_name_list:
       layers.append(self.model[name])
      
    if args.net_architect == 'DenseNet':
       layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
       num_out = 2048
    elif args.net_architect == 'DenseNet':
       num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
       num_out = 2048
    
    self.layer1 = nn.ModuleList(layers)
    self.flatten = nn.Flatten()
    self.prophazards = PropHazards(num_out, self.output_dim) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    
  def forward(self, x):
    for layer in self.layer1:
       x = layer(x)
    # print(f'x.size:{x.size()}')

    x = self.flatten(x)
    x = self.prophazards(x)
    
    return x

def get_output_shape(x, model):
  model.eval()
  x = model(x)
  return torch.tensor(x.shape[1:]).prod()#.cuda()

#%%
''' Loss Function '''

def cox_partial_likelihood(y_pred, y_true, n_intervals=19):
    # Get the number of samples
    num_samples = y_pred.size(0)
    
    # Extracting censoring and event information from y_true
    censored = 1. + y_true[:, 0:n_intervals] * (y_pred-1.)  # Censoring indicators
    events = 1. - y_true[:, n_intervals:2*n_intervals] * y_pred  # Event indicators
    
    # Calculating the risk scores
    risk_scores = torch.exp(y_pred)  # Using exponential as risk scores
    
    # Computing the log partial likelihood
    log_partial_likelihood = torch.zeros(num_samples, device=y_pred.device)
    for i in range(num_samples):
        risk_i = risk_scores[i]
        censored_i = censored[i]
        events_i = events[i]
        
        risk_sum = torch.sum(risk_scores[i:])
        events_sum = torch.sum(events[i:] * risk_scores[i:])
        
        log_partial_likelihood[i] = torch.log(events_sum / risk_sum)
    
    # Calculating the negative of the mean log partial likelihood
    loss = -torch.mean(log_partial_likelihood)
    
    return loss
  
def nnet_loss(y_pred, y_true, n_intervals = 19):
    
    ''' criterion argument의 순서 (y_pred, y_true 여야 돌아가고, y_true, y_pred 면 차원 안 맞음)가 중요하고 
    보통 (output, target 또는 label) 순으로 선언되며, 이 경우는 utils.py의 train_model 에 criterion(output, label) 로 되어 있음. '''
    
    cens_uncens = 1. + y_true[:, 0:n_intervals] * (y_pred-1.)
    
    uncens = 1. - y_true[:, n_intervals: 2 * n_intervals] * y_pred
    
    # print(f'y_pred.size:{y_pred.size()}') torch.size(10,19)
    # print(f'y_true.size:{y_true.size()}') torch.size(10,38)
    # print(f'cens_uncens:{cens_uncens.size()}') torch.size(10,19)
    # print(f'uncens.size:{uncens.size()}') torch.size(10,19)

    loss = torch.sum(-torch.log(torch.clip(torch.cat((cens_uncens, uncens), dim=-1), torch.finfo(torch.float32).eps, None)), axis=-1)
    # print(f'loss.size:{loss.size()}') # torch.size(10)
    loss = loss.mean()
    
    return loss

class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) # Real Data Distribution
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            target = target.data.unsqueeze(1).long() # Real Data Value
            true_dist.scatter_(1, target, self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        loss = self.lab_smooth(log_probs, labels)
        return loss

#%%
''' Training Function '''

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 100
    history = defaultdict(list)
    model = model.to(device)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                print(f'phase:{phase}')
                model.train() # Set model to training mode
            else:
                print(f'phase:{phase}')
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            # running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                 
                    loss = criterion(outputs, labels) # use this loss for any training statistics
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            # epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            # history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if phase=='valid' and epoch_loss <= best_loss: # epoch_acc >= best_acc:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                saved_model_path = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
                os.makedirs(saved_model_path, exist_ok=True)
                PATH = os.path.join(saved_model_path, f"{datetime.now().strftime('%d_%m_%H_%m')}_{args.net_architect}_epoch{epoch}.pth")
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Accuracy ",best_acc)
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
  
''' Training Function '''

def train_classification_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device,):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 100
    history = defaultdict(list)
    model = model.to(device)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if(phase == 'train'):
                print(f'phase:{phase}')
                model.train() # Set model to training mode
            else:
                print(f'phase:{phase}')
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            # running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                 
                    loss = criterion(outputs, labels) # use this loss for any training statistics
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            # epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            # history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if epoch_loss <= best_loss: # epoch_acc >= best_acc:
              # best_acc = epoch_acc
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())
              saved_model_path = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
              os.makedirs(saved_model_path, exist_ok=True)
              PATH = os.path.join(saved_model_path, f"{datetime.now().strftime('%d_%m_%H_%m')}_{args.net_architect}_epoch{epoch}.pth")
              torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Accuracy ",best_acc)
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

#%%

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

# BS at 1yr
def get_BS(event, duration, oneyr_survs, duration_set=365):
  y = [(evt, dur) for evt, dur in zip(np.asarray(event, dtype=bool), duration)]
  y = np.asarray(y, dtype=[('cens', bool), ('time', float)])
  times, score = brier_score(y, y, oneyr_survs, duration_set)
  print(f'BS score at {duration_set}:{score}')
  return score