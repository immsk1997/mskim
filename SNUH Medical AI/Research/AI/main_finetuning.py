'''Vision Transformer Fine-Tuning'''

'''Import API & Library'''
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

from utils import *
from attention_models import *
from vit_3d import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from medcam import medcam
from medcam import *

from skimage.transform import resize
from scipy import ndimage

'''Deep Learning HyperParameter, Computer Resource Setting'''
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123456) # 12347541
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') # 'death' # 
    parser.add_argument('--ext_dataset_name', type=str, default='UCSF') # 'TCGA' # 
    parser.add_argument('--dataset_list', nargs='+', default=['SNUH','UPenn','TCGA','severance'], help='selected_training_datasets')
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

#%%
'''Label Preparation & K-fold (10-fold)'''
df = save_label_dataset_list(main_args, args)
ext_df = save_label_ext_dataset(main_args, args)

combine_img(main_args, args)

df_proc_labels_train, event_train, duration_train = make_kfold_df_proc_labels(main_args,args, f'{args.dataset_name}', remove_idh_mut = main_args.remove_idh_mut)
df_proc_labels_test, event_test, duration_test = make_kfold_df_proc_labels(main_args,args, f'{main_args.ext_dataset_name}', remove_idh_mut = main_args.remove_idh_mut)

'''glioma class label (Pre-Trained)'''
df_class_labels_train = make_class_df(main_args,args, f'{args.dataset_name}', remove_idh_mut = main_args.remove_idh_mut)
df_class_labels_test = make_class_df(main_args,args, f'{main_args.ext_dataset_name}', remove_idh_mut = main_args.remove_idh_mut)

#%%
'''Train / Valid Model 설정'''

if args.net_architect == 'VisionTransformer' and args.finetuning_type == "classifier":
  model = vit_glioma_type_classifier(args=args).to(device)

'''Optimizer, Loss Function'''
base_optimizer = AdamP
optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2) 
scheduler = fetch_scheduler(optimizer)

'''Training (Internal DataSet)'''
if not main_args.save_grad_cam:
  model, history = train_classification(df_class_labels_train, args, model, criterion, optimizer, scheduler, device=device, fold=0, num_epochs=main_args.epochs)

  '''ViT Fine-Tuning'''
  pretrained_base = vit_glioma_type_classifier(args=args)
  pretrained_model = load_ckpt(args,pretrained_base)
    
  cls_extractor = ClsExtractor(pretrained_model)
  surv_pred_layer = nn.Linear(cls_extractor.embed_dim,args.n_intervals)
    
  class CustomViT(nn.Module):
    def __init__(self,cls_extractor,surv_pred_layer):
      super().__init__()
      self.cls_extractor = cls_extractor
      self.surv_pred_layer = surv_pred_layer
        
    def forward(self,x):
      _to_cls_layer = self.cls_extractor.forward_features(x)
      final_output = self.surv_pred_layer(_to_cls_layer)
        
      return torch.pow(torch.sigmoid(final_output), torch.exp(final_output))
      
  fine_tuning_model = CustomViT(cls_extractor,surv_pred_layer).to(device)
  fine_tuning_base_optimizer = AdamP
  fine_tuning_optimizer = SAM(fine_tuning_model.parameters(), fine_tuning_base_optimizer, lr=args.lr, weight_decay=args.weight_decay)

  fine_tuning_criterion = cox_partial_likelihood 

  fine_tuning_scheduler = fetch_scheduler(fine_tuning_optimizer)

  fine_tuning_model, history = run_fold_vit(df_proc_labels_train, args, fine_tuning_model, fine_tuning_criterion, fine_tuning_optimizer, fine_tuning_scheduler, device=device, fold=0, num_epochs=main_args.epochs)

# %%
''' Downstream Task : Survival Analysis & Grad_CAM (External DataSet) '''
''' grad_CAM, inference.py, train_inference.py in Same Working Directory : Internal (Train / Valid Set) 활용'''

test_gpu_id = main_args.test_gpu_id 
print(f'Testing on GPU {test_gpu_id}')

test_device = torch.device(test_gpu_id)

proc_label_path_test = os.path.join(args.proc_label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
df_proc_labels_test = pd.read_csv(proc_label_path_test, dtype='string')
df_proc_labels_test = df_proc_labels_test.set_index('ID')

if args.net_architect == "VisionTransformer" and args.finetuning_type =="classifier":
  fine_tuning_model = CustomViT(cls_extractor , surv_pred_layer).to(test_device)
  fine_tuning_model = load_ckpt(args, fine_tuning_model)
  test_data = ViTDataset(df = df_proc_labels_test, args = args, dataset_name=f'{main_args.ext_dataset_name}')
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

else:
  pass
#%%
'''Test Inference'''
df_DL_score_test = df_proc_labels_test.copy()
df_DL_score_test.drop(columns=df_DL_score_test.columns,inplace=True) 

for i in np.arange(n_intervals):
  df_DL_score_test.insert(int(i), f'MRI{i+1}', '')
df_DL_score_test.insert(n_intervals, 'oneyr_survs_test', '')

oneyr_survs_test = []
for subj_num, (inputs,labels) in enumerate(test_loader):
  fine_tuning_model.eval()
  inputs = inputs.to(test_device)
  labels = labels.to(test_device)

  y_pred = fine_tuning_model(inputs) # torch.Size([4, 19])
  print(f'y_pred.size:{y_pred.size()}')
  print(f'y_pred:{y_pred}')
  print(f'labels.size:{labels.size()}')
  print(f'labels:{labels}')
  print(f'subj_num:{subj_num}')
    
  ''' evaluate c-index (Survival Analysis) '''
  # ref : https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
  # ref : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221380929786

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