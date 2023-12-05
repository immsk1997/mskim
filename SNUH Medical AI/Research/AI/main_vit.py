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
import argparse

import math
from torchsummary import summary as summary

import torch # For building the networks 
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.nets import *

from utils_vit import *
from vit_3d import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from medcam import medcam
from medcam import *

from skimage.transform import resize
from scipy import ndimage

#%%
''' Setting '''
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123456) # 12347541
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') # 'death' # 
    parser.add_argument('--ext_dataset_name', type=str, default='severance') # 'TCGA' # 
    parser.add_argument('--dataset_list', nargs='+', default=['SNUH'], help='selected_training_datasets') # ,'TCGA'
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    return parser

#%%
main_args = get_args_parser().parse_args()

#%%
args = config()

breaks, n_intervals = get_n_intervals(fixed_interval_width = False) # True #
args.n_intervals = n_intervals

gpu_id = main_args.gpu_id

args.dataset_name = '_'.join(main_args.dataset_list)
print(f'Train dataset_name:{args.dataset_name}')

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

print(f'Training on GPU {gpu_id}')
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

#%%
os.environ['MKL_THREADING_LAYER'] = 'GNU' # in Linux, I had to write a script to call "export MKL_THREADING_LAYER=GNU" (which sets that environment variable) each time I activate the virtual environment, and a counter script to undo that change upon exiting the environment.
set_seed(main_args.seed)
print(f'Setting seed:{main_args.seed}')
#%%
to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
torch.cuda.empty_cache()
# print(get_dir(DATA_DIR))

#%%
get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))

df = save_label_dataset_list(main_args, args) # argument에 맞게 label DF 필터링하여 새롭게 저장
ext_df = save_label_ext_dataset(main_args, args) # argument에 맞게 label DF 필터링하여 새롭게 저장

#%%
'''여러개의 Dataset_list -> 하나의 Dataset name으로 통합'''
combine_img(main_args, args)

#%%
'''Label <-> Image Mapping, Fold 설정'''
df_proc_labels_train, event_train, duration_train = make_kfold_df_proc_labels(main_args, args, f'{args.dataset_name}', remove_idh_mut = main_args.remove_idh_mut)
df_proc_labels_test, event_test, duration_test = make_kfold_df_proc_labels(main_args, args, f'{main_args.ext_dataset_name}', remove_idh_mut = main_args.remove_idh_mut)

'''Model 설정'''
#%%
if args.net_architect == 'VisionTransformer':
  model = vit_gbm_patch16(args=args, num_classes=args.n_intervals).to(device)

print(f'Show Vision Transformer Architect:{vars(model)["_modules"]}')
layer_list = [n for n in vars(model)["_modules"]]
print(f'Vision Transformer Layer List:{layer_list}')

"""Optimizer, Loss Function"""
base_optimizer = AdamP
optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
criterion = TaylorCrossEntropyLoss(n=2,smoothing=0.2) #nnet_loss
scheduler = fetch_scheduler(optimizer)

'''Train / Validation'''
if not main_args.save_grad_cam:
  model, history = run_fold(df_proc_labels_train, args, model, criterion, optimizer, scheduler, device=device, fold=0, num_epochs=main_args.epochs)

# %%
test_gpu_id = main_args.test_gpu_id # int(main_args.gpu_id + 1) # 
print(f'Testing on GPU {test_gpu_id}')

test_device = torch.device(test_gpu_id)
model = vit_gbm_patch16(args=args, num_classes=args.n_intervals).to(test_device)
model = load_ckpt(args, model)

proc_label_path_test = os.path.join(args.proc_label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
df_proc_labels_test = pd.read_csv(proc_label_path_test, dtype='string')
df_proc_labels_test = df_proc_labels_test.set_index('ID')

test_data = ViTDataset(df = df_proc_labels_test, args = args, dataset_name=f'{main_args.ext_dataset_name}')
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

df_DL_score_test = df_proc_labels_test.copy()
df_DL_score_test.drop(columns=df_DL_score_test.columns,inplace=True) # index 열 제외하고 모두 삭제

for i in np.arange(n_intervals):
  df_DL_score_test.insert(int(i), f'MRI{i+1}', '')
df_DL_score_test.insert(n_intervals, 'oneyr_survs_test', '')

oneyr_survs_test = []
for subj_num, (inputs,labels) in enumerate(test_loader):
  model.eval()
  inputs = inputs.type(torch.cuda.FloatTensor)
  inputs = inputs.to(test_device)
  labels = labels.to(test_device)

  y_pred = model(inputs)
  print(f'y_pred:{y_pred}')
  print(f'labels:{labels}')
  print(f'subj_num:{subj_num}')
    
  ''' evaluate c-index (Survival Analysis) '''
  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
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

print(len(oneyr_survs_test)) # 132
oneyr_survs_test = np.array(oneyr_survs_test)

#%%
print(f'duration_test.shape:{duration_test.shape}')
print(f'oneyr_survs_test.shape:{oneyr_survs_test.shape}')
print(f'event_test.shape:{event_test.shape}')

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_test, oneyr_survs_test, event_test)

print(f'Original C-index for valid: {original_c_index:.4f}')
print(f'95% CI for C-index for valid: ({ci_lower:.4f}, {ci_upper:.4f})')

score_test = get_BS(event_test, duration_test, oneyr_survs_test)

#%%
'''Grad CAM'''
plt.switch_backend('agg')

test_img_path='/mnt/hdd3/mskim/GBL/data/severance/VIT/resized_BraTS/Sev001'
seqs = []
for seq in ['t1','t2','flair','t1ce']:
  seq=nib.load(os.path.join(test_img_path, f'{seq}_resized.nii.gz')).get_fdata() # _seg # _cropped
  print(f'seq range:min {seq.min()}-max {seq.max()}')
  # print(seq.shape)
  # torch.cat([x[sequence][tio.DATA] for sequence in self.SEQUENCE], axis=0)
  seqs.append(seq)

x = np.stack(seqs, axis=0)
x = torch.from_numpy(x)
x = torch.unsqueeze(x, axis=0)
print(f'x.shape:{x.shape}') # torch.Size([1, 4, 120, 120, 78])

z_slice_num = int(x.shape[-1]//2) # 39 # AXL
y_slice_num = int(x.shape[-2]//2) # 60 # SAG
x_slice_num = int(x.shape[-3]//2) # 60 # COR

seq_idx_dict = {'t1':0, 't2':1, 't1ce':2, 'flair':3}
selected_seq = 't1ce'
selected_seq_idx = seq_idx_dict[selected_seq]
print(f'selected_seq:{selected_seq}, {selected_seq_idx}')

#%%

print(f'args.attention_map_dir:{args.attention_map_dir}')

slice_3d = lambda x: x[selected_seq_idx,:,:,:]

rot_degree = 90

if main_args.save_grad_cam:
  cam_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
  cam_model = medcam.inject(model, backend='gcampp', output_dir="attention_maps", save_maps=True)

  superimposed_imgs = []
  cam_model.eval()

  for subj_num, batch in enumerate(cam_loader):
    batch = batch[0].to(test_device) # cuda()
    output = cam_model(batch)
    cam=cam_model.get_attention_map()
    print(type(cam)) # numpy array 
    print(f'cam.shape:{cam.shape}') # (1,1,4,4,3) # summary(model, (4, 120, 120, 78), device='cuda') 하면 나오는 shape이 (1,1,4,4,3) 임.
    print(f'input.shape:{batch.shape}') # torch.Size([1, 4, 120, 120, 78])

    subj_id = df_DL_score_test.index[subj_num]
    print(f'subj_id:{subj_id}')

    img_4d = batch.squeeze().cpu().numpy()
    img_3d = slice_3d(img_4d)

    img_3d_scaled = min_max_norm(img_3d)

    result_3d = cam.squeeze()

    print(f'img_3d.shape:{img_3d.shape}') # (120, 120, 78)
    print(f'result_3d.shape:{result_3d.shape}') # (4, 4, 3)
    
    superimposed_img_3d, result_3d_resized = superimpose_img(img_3d_scaled, result_3d)
    print(f'superimposed_img_3d.shape:{superimposed_img_3d.shape}')
    print(f'result_3d_resized.shape:{result_3d_resized.shape}')
    
    ''' axl '''
    img_2d = img_3d[:,:,z_slice_num]
    superimposed_img_2d = superimposed_img_3d[:,:,z_slice_num]
    result_2d_resized = result_3d_resized[:,:,z_slice_num]
    result_2d = result_3d[:,:,0]

    rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278

    plt_saved_loc = os.path.join(attention_map_dir, 'grad_CAM_2d_axl')
    os.makedirs(plt_saved_loc, exist_ok=True)
    plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_axl.jpg'), dpi=300)

    ''' sag '''
    img_2d = img_3d[x_slice_num,:,:]
    superimposed_img_2d = superimposed_img_3d[x_slice_num,:,:]
    result_2d_resized = result_3d_resized[x_slice_num,:,:]
    result_2d = result_3d[:,:,1]

    rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278

    plt_saved_loc = os.path.join(attention_map_dir, 'grad_CAM_2d_sag')
    os.makedirs(plt_saved_loc, exist_ok=True)
    plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_sag.jpg'), dpi=300)

    ''' cor '''
    img_2d = img_3d[:,y_slice_num,:]
    superimposed_img_2d = superimposed_img_3d[:,y_slice_num,:]
    result_2d_resized = result_3d_resized[:,y_slice_num,:]
    result_2d = result_3d[:,:,2]

    rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278

    plt_saved_loc = os.path.join(attention_map_dir, 'grad_CAM_2d_cor')
    os.makedirs(plt_saved_loc, exist_ok=True)
    plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_cor.jpg'), dpi=300)

    saved_loc = os.path.join(attention_map_dir, 'grad_CAM')
    os.makedirs(saved_loc, exist_ok=True)

    superimposed_img_2d = min_max_norm(superimposed_img_2d)
    superimposed_img_2d = (superimposed_img_2d * 255).astype(np.uint8)
    grad_heatmap = cv2.applyColorMap(superimposed_img_2d[...,0], cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_num}_cv_jet.jpg'), grad_heatmap)
    
    plt_saved_loc_3d = os.path.join(attention_map_dir, 'grad_CAM_3d')
    os.makedirs(plt_saved_loc_3d, exist_ok=True)