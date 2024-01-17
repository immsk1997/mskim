args:{'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'T_0': 10, 'lr': 0.0001, 'min_lr': 1e-06, 'weight_decay': 1e-05, 
'n_fold': 10, 'smoothing': 0.3, 'batch_size': 16, 'net_architect': 'VisionTransformer', 'finetuning_type': None, 
'compart_name': 'resized', 'sequence': ['t1', 't2', 't1ce', 'flair'], 
'data_key_list': ['sex', 'age', 'IDH', 'glioma_type', 'glioma_num', 'MGMT', 'GBL', 'EOR', 'duration_death', 
'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion'], 'root_dir': '/mnt/hdd3/mskim/GBL', 
'data_dir': '/mnt/hdd3/mskim/GBL/data', 'label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels', 
'proc_label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels/proc_labels', 'exp_dir': '/mnt/hdd3/mskim/GBL/code/experiment', 
'dataset_name': ''}
Using VisionTransformer
Train dataset_name:UCSF_UPenn_TCGA_SNUH
Training on GPU 0
Setting seed:123456
dataset:UCSF for training
df_dataset.shape:(500, 13)
dataset:UPenn for training
df_dataset.shape:(425, 13)
dataset:TCGA for training
df_dataset.shape:(160, 13)
dataset:SNUH for training
df_dataset.shape:(1013, 13)
df_label_dataset_list.shape:(2098, 13)
filtering before GBL; 2098 cases
filtering after GBL; 1529 cases
ext_df_dataset.shape:(132, 13)
filtering before GBL; 132 cases
filtering after GBL; 107 cases
df.index.values:1529
list_dataset: ['UCSF', 'UPenn', 'TCGA', 'SNUH']
split_dataset: UCSF
split_img_label_comm_list:367
split_dataset: UPenn
split_img_label_comm_list:383
split_dataset: TCGA
split_img_label_comm_list:59
split_dataset: SNUH
split_img_label_comm_list:668
img_label_comm_list:1477
dataset_name:UCSF_UPenn_TCGA_SNUH, 1477
UCSF_UPenn_TCGA_SNUH df.shape: (1477, 13)
df.index.values:107
img_label_comm_list:107
dataset_name:severance, 107
severance df.shape: (107, 13)
num of train_data: 1329
num of valid_data: 148
Epoch 1/100
----------
phase:train
train Loss: 2.5976
phase:valid
valid Loss: 2.5730

Epoch 2/100
----------
phase:train
train Loss: 2.4965
phase:valid
valid Loss: 2.5270

Epoch 3/100
----------
phase:train
train Loss: 2.4573
phase:valid
valid Loss: 2.4897

Epoch 4/100
----------
phase:train
train Loss: 2.4562
phase:valid
valid Loss: 2.4555

Epoch 5/100
----------
phase:train
train Loss: 2.4385
phase:valid
valid Loss: 2.4753

Epoch 6/100
----------
phase:train
train Loss: 2.4126
phase:valid
valid Loss: 2.4648

Epoch 7/100
----------
phase:train
train Loss: 2.3961
phase:valid
valid Loss: 2.4332

Epoch 8/100
----------
phase:train
train Loss: 2.3677
phase:valid
valid Loss: 2.4107

Epoch 9/100
----------
phase:train
train Loss: 2.3467
phase:valid
valid Loss: 2.4136

Epoch 10/100
----------
phase:train
train Loss: 2.3226
phase:valid
valid Loss: 2.4060

Epoch 11/100
----------
phase:train
train Loss: 2.3124
phase:valid
valid Loss: 2.4050

Epoch 12/100
----------
phase:train
train Loss: 2.3107
phase:valid
valid Loss: 2.4028

Epoch 13/100
----------
phase:train
train Loss: 2.3146
phase:valid
valid Loss: 2.4111

Epoch 14/100
----------
phase:train
train Loss: 2.3266
phase:valid
valid Loss: 2.4200

Epoch 15/100
----------
phase:train
train Loss: 2.3285
phase:valid
valid Loss: 2.4183

Epoch 16/100
----------
phase:train
train Loss: 2.3501
phase:valid
valid Loss: 2.3928

Epoch 17/100
----------
phase:train
train Loss: 2.3699
phase:valid
valid Loss: 2.4240

Epoch 18/100
----------
phase:train
train Loss: 2.3465
phase:valid
valid Loss: 2.4639

Epoch 19/100
----------
phase:train
train Loss: 2.3298
phase:valid
valid Loss: 2.4838

Epoch 20/100
----------
phase:train
train Loss: 2.3276
phase:valid
valid Loss: 2.5692

Epoch 21/100
----------
phase:train
train Loss: 2.2850
phase:valid
valid Loss: 2.5772

Epoch 22/100
----------
phase:train
train Loss: 2.2960
phase:valid
valid Loss: 2.5995

Epoch 23/100
----------
phase:train
train Loss: 2.2284
phase:valid
valid Loss: 2.5489

Epoch 24/100
----------
phase:train
train Loss: 2.1235
phase:valid
valid Loss: 2.5867

Epoch 25/100
----------
phase:train
train Loss: 1.9760
phase:valid
valid Loss: 2.6930

Epoch 26/100
----------
phase:train
train Loss: 1.8049
phase:valid
valid Loss: 2.8652

Epoch 27/100
----------
phase:train
train Loss: 1.5764
phase:valid
valid Loss: 3.0666

Epoch 28/100
----------
phase:train
train Loss: 1.3176
phase:valid
valid Loss: 3.1917

Epoch 29/100
----------
phase:train
train Loss: 1.1041
phase:valid
valid Loss: 3.1940

Epoch 30/100
----------
phase:train
train Loss: 0.9868
phase:valid
valid Loss: 3.2396

Epoch 31/100
----------
phase:train
train Loss: 0.9439
phase:valid
valid Loss: 3.2464

Epoch 32/100
----------
phase:train
train Loss: 0.9354
phase:valid
valid Loss: 3.2534

Epoch 33/100
----------
phase:train
train Loss: 0.9173
phase:valid
valid Loss: 3.2882

Epoch 34/100
----------
phase:train
train Loss: 0.8877
phase:valid
valid Loss: 3.3348

Epoch 35/100
----------
phase:train
train Loss: 0.8560
phase:valid
valid Loss: 3.4859

Epoch 36/100
----------
phase:train
train Loss: 0.9132
phase:valid
valid Loss: 3.4367

Epoch 37/100
----------
phase:train
train Loss: 1.0393
phase:valid
valid Loss: 3.5904

Epoch 38/100
----------
phase:train
train Loss: 1.1098
phase:valid
valid Loss: 3.5228

Epoch 39/100
----------
phase:train
train Loss: 1.0665
phase:valid
valid Loss: 3.3966

Epoch 40/100
----------
phase:train
train Loss: 0.8838
phase:valid
valid Loss: 3.5245

Epoch 41/100
----------
phase:train
train Loss: 0.9179
phase:valid
valid Loss: 3.7720

Epoch 42/100
----------
phase:train
train Loss: 0.7026
phase:valid
valid Loss: 3.8286

Epoch 43/100
----------
phase:train
train Loss: 0.5837
phase:valid
valid Loss: 3.9451

Epoch 44/100
----------
phase:train
train Loss: 0.4958
phase:valid
valid Loss: 4.0327

Epoch 45/100
----------
phase:train
train Loss: 0.4382
phase:valid
valid Loss: 4.1577

Epoch 46/100
----------
phase:train
train Loss: 0.4080
phase:valid
valid Loss: 4.1136

Epoch 47/100
----------
phase:train
train Loss: 0.3929
phase:valid
valid Loss: 4.1422

Epoch 48/100
----------
phase:train
train Loss: 0.3859
phase:valid
valid Loss: 4.2131

Epoch 49/100
----------
phase:train
train Loss: 0.3835
phase:valid
valid Loss: 4.2308

Epoch 50/100
----------
phase:train
train Loss: 0.3826
phase:valid
valid Loss: 4.2330

Epoch 51/100
----------
phase:train
train Loss: 0.3824
phase:valid
valid Loss: 4.2334

Epoch 52/100
----------
phase:train
train Loss: 0.3823
phase:valid
valid Loss: 4.2354

Epoch 53/100
----------
phase:train
train Loss: 0.3820
phase:valid
valid Loss: 4.2350

Epoch 54/100
----------
phase:train
train Loss: 0.3811
phase:valid
valid Loss: 4.2594

Epoch 55/100
----------
phase:train
train Loss: 0.3804
phase:valid
valid Loss: 4.2773

Epoch 56/100
----------
phase:train
train Loss: 0.3797
phase:valid
valid Loss: 4.3278

Epoch 57/100
----------
phase:train
train Loss: 0.4634
phase:valid
valid Loss: 4.1295

Epoch 58/100
----------
phase:train
train Loss: 0.4857
phase:valid
valid Loss: 4.1824

Epoch 59/100
----------
phase:train
train Loss: 0.6801
phase:valid
valid Loss: 4.2893

Epoch 60/100
----------
phase:train
train Loss: 0.6238
phase:valid
valid Loss: 4.5225

Epoch 61/100
----------
phase:train
train Loss: 0.5931
phase:valid
valid Loss: 4.3798

Epoch 62/100
----------
phase:train
train Loss: 0.7381
phase:valid
valid Loss: 4.4162

Epoch 63/100
----------
phase:train
train Loss: 0.5564
phase:valid
valid Loss: 4.5135

Epoch 64/100
----------
phase:train
train Loss: 0.4750
phase:valid
valid Loss: 4.6044

Epoch 65/100
----------
phase:train
train Loss: 0.4056
phase:valid
valid Loss: 4.6644

Epoch 66/100
----------
phase:train
train Loss: 0.3800
phase:valid
valid Loss: 4.6862

Epoch 67/100
----------
phase:train
train Loss: 0.3740
phase:valid
valid Loss: 4.6979

Epoch 68/100
----------
phase:train
train Loss: 0.3714
phase:valid
valid Loss: 4.7534

Epoch 69/100
----------
phase:train
train Loss: 0.3705
phase:valid
valid Loss: 4.7656

Epoch 70/100
----------
phase:train
train Loss: 0.3702
phase:valid
valid Loss: 4.7703

Epoch 71/100
----------
phase:train
train Loss: 0.3702
phase:valid
valid Loss: 4.7719

Epoch 72/100
----------
phase:train
train Loss: 0.3701
phase:valid
valid Loss: 4.7743

Epoch 73/100
----------
phase:train
train Loss: 0.3700
phase:valid
valid Loss: 4.7802

Epoch 74/100
----------
phase:train
train Loss: 0.3696
phase:valid
valid Loss: 4.8003

Epoch 75/100
----------
phase:train
train Loss: 0.3692
phase:valid
valid Loss: 4.8330

Epoch 76/100
----------
phase:train
train Loss: 0.3686
phase:valid
valid Loss: 4.8618

Epoch 77/100
----------
phase:train
train Loss: 0.3678
phase:valid
valid Loss: 4.8939

Epoch 78/100
----------
phase:train
train Loss: 0.3684
phase:valid
valid Loss: 4.8149

Epoch 79/100
----------
phase:train
train Loss: 0.4033
phase:valid
valid Loss: 4.3974

Epoch 80/100
----------
phase:train
train Loss: 0.9739
phase:valid
valid Loss: 4.2024

Epoch 81/100
----------
phase:train
train Loss: 0.8752
phase:valid
valid Loss: 4.3610

Epoch 82/100
----------
phase:train
train Loss: 0.5827
phase:valid
valid Loss: 4.4065

Epoch 83/100
----------
phase:train
train Loss: 0.4975
phase:valid
valid Loss: 4.6197

Epoch 84/100
----------
phase:train
train Loss: 0.4113
phase:valid
valid Loss: 4.7369

Epoch 85/100
----------
phase:train
train Loss: 0.3875
phase:valid
valid Loss: 4.8207

Epoch 86/100
----------
phase:train
train Loss: 0.3751
phase:valid
valid Loss: 4.9314

Epoch 87/100
----------
phase:train
train Loss: 0.3733
phase:valid
valid Loss: 4.9895

Epoch 88/100
----------
phase:train
train Loss: 0.3697
phase:valid
valid Loss: 5.0209

Epoch 89/100
----------
phase:train
train Loss: 0.3690
phase:valid
valid Loss: 5.0417

Epoch 90/100
----------
phase:train
train Loss: 0.3688
phase:valid
valid Loss: 5.0468

Epoch 91/100
----------
phase:train
train Loss: 0.3688
phase:valid
valid Loss: 5.0489

Epoch 92/100
----------
phase:train
train Loss: 0.3687
phase:valid
valid Loss: 5.0536

Epoch 93/100
----------
phase:train
train Loss: 0.3686
phase:valid
valid Loss: 5.0678

Epoch 94/100
----------
phase:train
train Loss: 0.3683
phase:valid
valid Loss: 5.0860

Epoch 95/100
----------
phase:train
train Loss: 0.3680
phase:valid
valid Loss: 5.1191

Epoch 96/100
----------
phase:train
train Loss: 0.3675
phase:valid
valid Loss: 5.1727

Epoch 97/100
----------
phase:train
train Loss: 0.3669
phase:valid
valid Loss: 5.1990

Epoch 98/100
----------
phase:train
train Loss: 0.3665
phase:valid
valid Loss: 5.2560

Epoch 99/100
----------
phase:train
train Loss: 0.3662
phase:valid
valid Loss: 5.3169

Epoch 100/100
----------
phase:train
train Loss: 0.3659
phase:valid
valid Loss: 5.3673

Training complete in 7h 43m 31s
Best Loss  2.39276123046875
Testing on GPU 1
latest_ckpt_model: /mnt/hdd3/mskim/GBL/code/saved_models/VisionTransformer/04_01_23_01_VisionTransformer_epoch16.pth
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9562, 0.9313, 0.9095, 0.8893, 0.9386, 0.8610, 0.8160, 0.8215, 0.6360,
         0.7132, 0.7768, 0.8685, 0.8049, 0.6949, 0.8218, 0.7893, 0.6516, 0.6359,
         0.8326]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:0
oneyr_surv_test: [0.5820221]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9908, 0.9682, 0.9638, 0.9538, 0.9424, 0.9194, 0.8760, 0.9013, 0.8966,
         0.8781, 0.8511, 0.9327, 0.8517, 0.8111, 0.8513, 0.7810, 0.7353, 0.7549,
         0.7878]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:1
oneyr_surv_test: [0.76407534]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9855, 0.9642, 0.9414, 0.9460, 0.9449, 0.9185, 0.8403, 0.8074, 0.8921,
         0.7580, 0.8484, 0.9247, 0.8779, 0.6945, 0.7580, 0.7905, 0.5884, 0.7285,
         0.8276]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:2
oneyr_surv_test: [0.7345127]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9643, 0.9392, 0.9244, 0.8958, 0.9389, 0.8684, 0.8325, 0.8454, 0.6570,
         0.7562, 0.7910, 0.8680, 0.8015, 0.7250, 0.8392, 0.7910, 0.6841, 0.6495,
         0.8263]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:3
oneyr_surv_test: [0.6114147]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9647, 0.9393, 0.9262, 0.8907, 0.9402, 0.8737, 0.8286, 0.8494, 0.6530,
         0.7577, 0.7891, 0.8680, 0.8016, 0.7229, 0.8410, 0.7900, 0.6854, 0.6527,
         0.8274]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:4
oneyr_surv_test: [0.6140211]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9909, 0.9684, 0.9651, 0.9522, 0.9406, 0.9180, 0.8761, 0.9049, 0.8851,
         0.8879, 0.8469, 0.9313, 0.8394, 0.8191, 0.8650, 0.7838, 0.7557, 0.7552,
         0.7840]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:5
oneyr_surv_test: [0.7614306]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9532, 0.9253, 0.8940, 0.8898, 0.9318, 0.8493, 0.8132, 0.8049, 0.6212,
         0.6949, 0.7691, 0.8684, 0.8038, 0.6883, 0.8106, 0.7895, 0.6343, 0.6243,
         0.8347]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:6
oneyr_surv_test: [0.5552013]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9821, 0.9610, 0.9430, 0.9414, 0.9417, 0.9052, 0.8404, 0.8108, 0.8572,
         0.7554, 0.8436, 0.9016, 0.8587, 0.7031, 0.7887, 0.7919, 0.6211, 0.7096,
         0.8312]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:7
oneyr_surv_test: [0.7141728]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9729, 0.9499, 0.9210, 0.9238, 0.9368, 0.8889, 0.8077, 0.7683, 0.7825,
         0.7161, 0.8071, 0.8963, 0.8385, 0.6615, 0.7817, 0.7936, 0.5922, 0.6770,
         0.8344]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:8
oneyr_surv_test: [0.6547461]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9905, 0.9679, 0.9648, 0.9513, 0.9407, 0.9167, 0.8740, 0.9028, 0.8781,
         0.8870, 0.8446, 0.9299, 0.8346, 0.8177, 0.8685, 0.7846, 0.7590, 0.7535,
         0.7851]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:9
oneyr_surv_test: [0.7587944]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9843, 0.9636, 0.9501, 0.9354, 0.9414, 0.9168, 0.8258, 0.8212, 0.8445,
         0.7854, 0.8424, 0.9066, 0.8496, 0.7095, 0.8128, 0.7950, 0.6467, 0.7291,
         0.8207]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:10
oneyr_surv_test: [0.72754824]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9609, 0.9298, 0.8990, 0.8884, 0.9352, 0.8601, 0.8137, 0.8094, 0.6088,
         0.7215, 0.7717, 0.8683, 0.7950, 0.6888, 0.8246, 0.7939, 0.6434, 0.6306,
         0.8321]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:11
oneyr_surv_test: [0.5739341]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9864, 0.9668, 0.9531, 0.9477, 0.9467, 0.9226, 0.8500, 0.8387, 0.9017,
         0.7814, 0.8606, 0.9171, 0.8788, 0.7233, 0.7800, 0.7831, 0.6244, 0.7354,
         0.8237]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:12
oneyr_surv_test: [0.7522944]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9686, 0.9450, 0.9341, 0.9015, 0.9411, 0.8787, 0.8304, 0.8517, 0.6816,
         0.7703, 0.7970, 0.8727, 0.8007, 0.7303, 0.8470, 0.7898, 0.6930, 0.6661,
         0.8225]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], device='cuda:1')
subj_num:13
oneyr_surv_test: [0.63749987]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9888, 0.9683, 0.9591, 0.9547, 0.9425, 0.9192, 0.8695, 0.8732, 0.9136,
         0.8226, 0.8669, 0.9249, 0.8763, 0.7670, 0.8005, 0.7745, 0.6622, 0.7415,
         0.8086]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:14
oneyr_surv_test: [0.75948405]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9829, 0.9565, 0.9515, 0.9242, 0.9370, 0.8989, 0.8477, 0.8770, 0.7309,
         0.8553, 0.8049, 0.9012, 0.7846, 0.7846, 0.8836, 0.7993, 0.7620, 0.7106,
         0.8034]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:15
oneyr_surv_test: [0.69627935]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9839, 0.9630, 0.9521, 0.9422, 0.9415, 0.9089, 0.8524, 0.8498, 0.8698,
         0.7868, 0.8611, 0.9063, 0.8594, 0.7374, 0.8061, 0.7819, 0.6537, 0.7196,
         0.8209]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:16
oneyr_surv_test: [0.7273901]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9734, 0.9503, 0.9324, 0.9274, 0.9390, 0.8859, 0.8310, 0.8187, 0.7953,
         0.7286, 0.8292, 0.8865, 0.8410, 0.6999, 0.7955, 0.7880, 0.6312, 0.6767,
         0.8335]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:17
oneyr_surv_test: [0.66542476]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9704, 0.9458, 0.9246, 0.9124, 0.9353, 0.8811, 0.8165, 0.8064, 0.7324,
         0.7337, 0.8077, 0.8793, 0.8232, 0.6882, 0.8051, 0.7915, 0.6338, 0.6618,
         0.8337]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:18
oneyr_surv_test: [0.6380011]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9728, 0.9510, 0.9417, 0.9137, 0.9414, 0.8867, 0.8317, 0.8479, 0.7396,
         0.7737, 0.8186, 0.8797, 0.8144, 0.7278, 0.8403, 0.7885, 0.6857, 0.6823,
         0.8209]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:19
oneyr_surv_test: [0.6644707]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9892, 0.9687, 0.9581, 0.9540, 0.9450, 0.9253, 0.8634, 0.8568, 0.9196,
         0.8043, 0.8718, 0.9219, 0.8838, 0.7452, 0.7784, 0.7804, 0.6357, 0.7445,
         0.8191]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:20
oneyr_surv_test: [0.76589596]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9897, 0.9696, 0.9587, 0.9549, 0.9437, 0.9244, 0.8657, 0.8625, 0.9222,
         0.8086, 0.8717, 0.9257, 0.8853, 0.7535, 0.7824, 0.7761, 0.6380, 0.7479,
         0.8143]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:21
oneyr_surv_test: [0.76632535]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9773, 0.9527, 0.9455, 0.9240, 0.9383, 0.8835, 0.8507, 0.8621, 0.7425,
         0.8154, 0.8130, 0.8902, 0.8008, 0.7591, 0.8582, 0.7933, 0.7220, 0.6894,
         0.8135]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:22
oneyr_surv_test: [0.6743346]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9873, 0.9670, 0.9558, 0.9487, 0.9441, 0.9193, 0.8464, 0.8393, 0.8848,
         0.8141, 0.8570, 0.9191, 0.8620, 0.7314, 0.8068, 0.7883, 0.6520, 0.7410,
         0.8158]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:23
oneyr_surv_test: [0.75139594]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9628, 0.9345, 0.9171, 0.8835, 0.9364, 0.8683, 0.8251, 0.8398, 0.6233,
         0.7493, 0.7824, 0.8630, 0.7980, 0.7155, 0.8368, 0.7917, 0.6766, 0.6407,
         0.8284]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:24
oneyr_surv_test: [0.5926941]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9779, 0.9575, 0.9459, 0.9276, 0.9385, 0.8975, 0.8257, 0.8332, 0.7844,
         0.7855, 0.8269, 0.8961, 0.8220, 0.7200, 0.8317, 0.7939, 0.6718, 0.7003,
         0.8200]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:25
oneyr_surv_test: [0.6919058]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9449, 0.9115, 0.8678, 0.8756, 0.9300, 0.8338, 0.8150, 0.8005, 0.5555,
         0.6956, 0.7346, 0.8663, 0.7921, 0.6847, 0.8171, 0.7900, 0.6341, 0.6053,
         0.8355]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:26
oneyr_surv_test: [0.50748014]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9626, 0.9391, 0.9172, 0.9000, 0.9381, 0.8769, 0.8084, 0.8098, 0.6790,
         0.7218, 0.7884, 0.8741, 0.8125, 0.6869, 0.8134, 0.7919, 0.6413, 0.6486,
         0.8329]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:27
oneyr_surv_test: [0.61385846]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9678, 0.9407, 0.9276, 0.8929, 0.9402, 0.8793, 0.8296, 0.8495, 0.6515,
         0.7746, 0.7867, 0.8724, 0.7976, 0.7285, 0.8487, 0.7909, 0.6939, 0.6567,
         0.8243]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:28
oneyr_surv_test: [0.6233555]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9893, 0.9672, 0.9638, 0.9460, 0.9406, 0.9160, 0.8694, 0.8900, 0.8752,
         0.8651, 0.8592, 0.9145, 0.8441, 0.7904, 0.8500, 0.7873, 0.7324, 0.7452,
         0.7994]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:29
oneyr_surv_test: [0.75171095]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9768, 0.9556, 0.9411, 0.9273, 0.9386, 0.8927, 0.8323, 0.8253, 0.8119,
         0.7429, 0.8408, 0.8837, 0.8426, 0.7068, 0.8023, 0.7895, 0.6400, 0.6907,
         0.8300]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:30
oneyr_surv_test: [0.68256]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9591, 0.9351, 0.9141, 0.8938, 0.9387, 0.8674, 0.8128, 0.8163, 0.6581,
         0.7201, 0.7848, 0.8721, 0.8069, 0.6932, 0.8197, 0.7909, 0.6483, 0.6436,
         0.8302]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:31
oneyr_surv_test: [0.596628]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9874, 0.9656, 0.9560, 0.9526, 0.9416, 0.9093, 0.8657, 0.8683, 0.8743,
         0.8432, 0.8454, 0.9263, 0.8486, 0.7796, 0.8394, 0.7847, 0.6941, 0.7376,
         0.8065]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:32
oneyr_surv_test: [0.7434714]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9624, 0.9346, 0.9066, 0.8998, 0.9350, 0.8677, 0.8161, 0.8066, 0.6607,
         0.7247, 0.7727, 0.8733, 0.8096, 0.6898, 0.8142, 0.7937, 0.6393, 0.6411,
         0.8340]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:33
oneyr_surv_test: [0.5951955]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9812, 0.9605, 0.9488, 0.9325, 0.9431, 0.9097, 0.8346, 0.8338, 0.8492,
         0.7582, 0.8560, 0.8935, 0.8561, 0.7109, 0.8003, 0.7856, 0.6393, 0.7111,
         0.8274]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:34
oneyr_surv_test: [0.71536344]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9904, 0.9689, 0.9659, 0.9520, 0.9405, 0.9165, 0.8740, 0.8984, 0.8879,
         0.8735, 0.8553, 0.9268, 0.8473, 0.8101, 0.8578, 0.7805, 0.7400, 0.7543,
         0.7902]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:35
oneyr_surv_test: [0.7605387]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9596, 0.9287, 0.9064, 0.8793, 0.9342, 0.8590, 0.8269, 0.8388, 0.5817,
         0.7519, 0.7576, 0.8686, 0.7859, 0.7180, 0.8422, 0.7954, 0.6807, 0.6316,
         0.8283]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:36
oneyr_surv_test: [0.5699531]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9728, 0.9505, 0.9319, 0.9277, 0.9365, 0.8795, 0.8350, 0.8255, 0.7800,
         0.7450, 0.8244, 0.8888, 0.8313, 0.7142, 0.8091, 0.7885, 0.6427, 0.6755,
         0.8279]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:37
oneyr_surv_test: [0.6584882]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9904, 0.9692, 0.9641, 0.9523, 0.9414, 0.9196, 0.8722, 0.8890, 0.9019,
         0.8633, 0.8639, 0.9238, 0.8591, 0.7929, 0.8367, 0.7801, 0.7131, 0.7527,
         0.7951]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:38
oneyr_surv_test: [0.7630021]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9749, 0.9499, 0.9366, 0.9252, 0.9370, 0.8756, 0.8497, 0.8548, 0.7293,
         0.8063, 0.8024, 0.8954, 0.7979, 0.7567, 0.8558, 0.7918, 0.7067, 0.6784,
         0.8157]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:39
oneyr_surv_test: [0.6583971]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9870, 0.9646, 0.9621, 0.9405, 0.9397, 0.9079, 0.8674, 0.8933, 0.8423,
         0.8643, 0.8461, 0.9097, 0.8262, 0.7978, 0.8671, 0.7867, 0.7513, 0.7343,
         0.7983]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:40
oneyr_surv_test: [0.7350466]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9673, 0.9460, 0.9314, 0.9078, 0.9407, 0.8785, 0.8240, 0.8335, 0.7197,
         0.7420, 0.8059, 0.8770, 0.8156, 0.7082, 0.8264, 0.7884, 0.6623, 0.6668,
         0.8278]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:41
oneyr_surv_test: [0.6393404]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9613, 0.9319, 0.9133, 0.8848, 0.9367, 0.8627, 0.8300, 0.8413, 0.6236,
         0.7442, 0.7778, 0.8637, 0.8008, 0.7152, 0.8337, 0.7914, 0.6736, 0.6367,
         0.8309]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:42
oneyr_surv_test: [0.58503854]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9645, 0.9397, 0.9140, 0.9067, 0.9366, 0.8740, 0.8071, 0.7935, 0.7008,
         0.7199, 0.7917, 0.8809, 0.8169, 0.6764, 0.8040, 0.7923, 0.6254, 0.6514,
         0.8342]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:43
oneyr_surv_test: [0.61486924]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9633, 0.9371, 0.9225, 0.8875, 0.9407, 0.8746, 0.8257, 0.8464, 0.6300,
         0.7575, 0.7812, 0.8673, 0.7955, 0.7221, 0.8445, 0.7906, 0.6874, 0.6484,
         0.8266]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], device='cuda:1')
subj_num:44
oneyr_surv_test: [0.6079817]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9760, 0.9532, 0.9428, 0.9152, 0.9397, 0.8943, 0.8300, 0.8443, 0.7418,
         0.7926, 0.8144, 0.8870, 0.8118, 0.7248, 0.8419, 0.7914, 0.6881, 0.6866,
         0.8216]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:45
oneyr_surv_test: [0.6745022]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9626, 0.9365, 0.9205, 0.8850, 0.9395, 0.8720, 0.8243, 0.8461, 0.6232,
         0.7511, 0.7761, 0.8654, 0.7936, 0.7194, 0.8430, 0.7904, 0.6826, 0.6466,
         0.8267]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:46
oneyr_surv_test: [0.6016497]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9829, 0.9582, 0.9536, 0.9260, 0.9383, 0.9000, 0.8498, 0.8840, 0.7454,
         0.8511, 0.8100, 0.8992, 0.7914, 0.7903, 0.8836, 0.7967, 0.7635, 0.7147,
         0.8029]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:47
oneyr_surv_test: [0.7022379]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9889, 0.9667, 0.9646, 0.9473, 0.9403, 0.9129, 0.8701, 0.8948, 0.8666,
         0.8733, 0.8528, 0.9171, 0.8343, 0.8038, 0.8644, 0.7842, 0.7500, 0.7446,
         0.7926]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:48
oneyr_surv_test: [0.7498179]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9792, 0.9585, 0.9513, 0.9269, 0.9418, 0.8983, 0.8416, 0.8574, 0.8009,
         0.7965, 0.8360, 0.8910, 0.8286, 0.7421, 0.8408, 0.7878, 0.6937, 0.7061,
         0.8169]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:49
oneyr_surv_test: [0.70011276]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9606, 0.9372, 0.9186, 0.8987, 0.9398, 0.8695, 0.8187, 0.8238, 0.6739,
         0.7261, 0.7925, 0.8737, 0.8103, 0.7001, 0.8228, 0.7883, 0.6556, 0.6488,
         0.8299]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:50
oneyr_surv_test: [0.607318]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9898, 0.9692, 0.9608, 0.9519, 0.9439, 0.9256, 0.8645, 0.8652, 0.9175,
         0.8202, 0.8748, 0.9188, 0.8798, 0.7526, 0.7906, 0.7804, 0.6541, 0.7480,
         0.8125]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], device='cuda:1')
subj_num:51
oneyr_surv_test: [0.7664186]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9884, 0.9685, 0.9592, 0.9495, 0.9427, 0.9220, 0.8601, 0.8609, 0.9005,
         0.8227, 0.8690, 0.9191, 0.8678, 0.7496, 0.8039, 0.7840, 0.6603, 0.7411,
         0.8114]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:52
oneyr_surv_test: [0.7578025]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9622, 0.9402, 0.9227, 0.9001, 0.9400, 0.8719, 0.8171, 0.8246, 0.6874,
         0.7267, 0.7933, 0.8752, 0.8114, 0.6992, 0.8228, 0.7883, 0.6535, 0.6545,
         0.8288]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:53
oneyr_surv_test: [0.6157448]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9842, 0.9645, 0.9445, 0.9438, 0.9445, 0.9169, 0.8335, 0.8013, 0.8838,
         0.7555, 0.8524, 0.9160, 0.8718, 0.6920, 0.7697, 0.7883, 0.5958, 0.7254,
         0.8274]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:54
oneyr_surv_test: [0.73288846]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9789, 0.9566, 0.9486, 0.9287, 0.9412, 0.8952, 0.8479, 0.8571, 0.7942,
         0.8108, 0.8359, 0.8925, 0.8234, 0.7468, 0.8423, 0.7881, 0.7026, 0.7000,
         0.8158]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:55
oneyr_surv_test: [0.6951054]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9812, 0.9596, 0.9535, 0.9269, 0.9420, 0.9072, 0.8396, 0.8615, 0.8009,
         0.8088, 0.8413, 0.8911, 0.8286, 0.7453, 0.8434, 0.7875, 0.6998, 0.7095,
         0.8162]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:56
oneyr_surv_test: [0.7111414]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9547, 0.9198, 0.8824, 0.8801, 0.9290, 0.8505, 0.8081, 0.7988, 0.5807,
         0.7153, 0.7415, 0.8707, 0.7960, 0.6839, 0.8180, 0.7948, 0.6392, 0.6207,
         0.8327]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:57
oneyr_surv_test: [0.53881544]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9824, 0.9611, 0.9460, 0.9446, 0.9403, 0.8983, 0.8523, 0.8377, 0.8458,
         0.7917, 0.8401, 0.9094, 0.8470, 0.7376, 0.8170, 0.7873, 0.6562, 0.7113,
         0.8224]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:58
oneyr_surv_test: [0.7127081]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9739, 0.9512, 0.9306, 0.9215, 0.9374, 0.8862, 0.8212, 0.8067, 0.7695,
         0.7495, 0.8133, 0.8930, 0.8271, 0.6908, 0.8089, 0.7933, 0.6295, 0.6801,
         0.8279]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:59
oneyr_surv_test: [0.6599463]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9580, 0.9319, 0.9111, 0.8863, 0.9380, 0.8631, 0.8258, 0.8427, 0.5957,
         0.7534, 0.7601, 0.8735, 0.7837, 0.7217, 0.8466, 0.7920, 0.6864, 0.6376,
         0.8251]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:60
oneyr_surv_test: [0.5836684]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9592, 0.9355, 0.9154, 0.8961, 0.9398, 0.8682, 0.8181, 0.8220, 0.6633,
         0.7234, 0.7891, 0.8730, 0.8085, 0.6981, 0.8223, 0.7889, 0.6545, 0.6446,
         0.8303]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:61
oneyr_surv_test: [0.60054815]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9848, 0.9650, 0.9524, 0.9404, 0.9436, 0.9197, 0.8364, 0.8296, 0.8728,
         0.7845, 0.8599, 0.9068, 0.8618, 0.7124, 0.7957, 0.7862, 0.6345, 0.7261,
         0.8229]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:62
oneyr_surv_test: [0.7386293]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9622, 0.9363, 0.9205, 0.8857, 0.9397, 0.8737, 0.8220, 0.8408, 0.6264,
         0.7511, 0.7775, 0.8682, 0.7964, 0.7137, 0.8391, 0.7904, 0.6794, 0.6448,
         0.8283]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:63
oneyr_surv_test: [0.6030475]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9498, 0.9197, 0.8920, 0.8636, 0.9353, 0.8511, 0.8168, 0.8303, 0.5577,
         0.7073, 0.7456, 0.8595, 0.7883, 0.6982, 0.8299, 0.7909, 0.6564, 0.6183,
         0.8324]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:64
oneyr_surv_test: [0.5356798]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9901, 0.9706, 0.9633, 0.9509, 0.9424, 0.9240, 0.8669, 0.8724, 0.9162,
         0.8243, 0.8779, 0.9167, 0.8778, 0.7609, 0.7994, 0.7783, 0.6647, 0.7506,
         0.8107]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:65
oneyr_surv_test: [0.7664279]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9677, 0.9419, 0.9297, 0.8922, 0.9405, 0.8792, 0.8295, 0.8527, 0.6483,
         0.7720, 0.7856, 0.8700, 0.7945, 0.7304, 0.8503, 0.7916, 0.6968, 0.6589,
         0.8239]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:66
oneyr_surv_test: [0.6252252]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9768, 0.9568, 0.9366, 0.9324, 0.9422, 0.9015, 0.8201, 0.7966, 0.8305,
         0.7256, 0.8355, 0.8974, 0.8520, 0.6848, 0.7843, 0.7905, 0.6085, 0.6966,
         0.8318]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:67
oneyr_surv_test: [0.6932201]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9843, 0.9615, 0.9570, 0.9339, 0.9414, 0.9067, 0.8556, 0.8855, 0.8090,
         0.8532, 0.8287, 0.9096, 0.8154, 0.7880, 0.8716, 0.7887, 0.7494, 0.7258,
         0.8006]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:68
oneyr_surv_test: [0.7219981]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9870, 0.9671, 0.9578, 0.9423, 0.9433, 0.9204, 0.8466, 0.8521, 0.8724,
         0.8218, 0.8563, 0.9169, 0.8536, 0.7392, 0.8217, 0.7883, 0.6670, 0.7395,
         0.8107]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], device='cuda:1')
subj_num:69
oneyr_surv_test: [0.74798304]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9859, 0.9666, 0.9460, 0.9468, 0.9449, 0.9215, 0.8381, 0.8051, 0.8955,
         0.7721, 0.8474, 0.9237, 0.8762, 0.6969, 0.7653, 0.7887, 0.5934, 0.7342,
         0.8238]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:70
oneyr_surv_test: [0.7432495]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9568, 0.9316, 0.9091, 0.8914, 0.9382, 0.8627, 0.8151, 0.8176, 0.6437,
         0.7148, 0.7806, 0.8716, 0.8062, 0.6944, 0.8206, 0.7892, 0.6499, 0.6380,
         0.8316]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:71
oneyr_surv_test: [0.5845758]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9865, 0.9669, 0.9580, 0.9427, 0.9444, 0.9201, 0.8454, 0.8510, 0.8736,
         0.8161, 0.8565, 0.9158, 0.8549, 0.7382, 0.8193, 0.7883, 0.6637, 0.7393,
         0.8110]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:72
oneyr_surv_test: [0.74846727]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9567, 0.9318, 0.9120, 0.8833, 0.9395, 0.8669, 0.8170, 0.8348, 0.6152,
         0.7262, 0.7727, 0.8633, 0.7979, 0.7065, 0.8320, 0.7890, 0.6661, 0.6370,
         0.8301]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:73
oneyr_surv_test: [0.58486456]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9565, 0.9324, 0.9108, 0.8890, 0.9383, 0.8619, 0.8136, 0.8205, 0.6406,
         0.7105, 0.7753, 0.8687, 0.8046, 0.6920, 0.8212, 0.7893, 0.6486, 0.6383,
         0.8325]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:74
oneyr_surv_test: [0.58401287]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9829, 0.9623, 0.9509, 0.9413, 0.9388, 0.9038, 0.8483, 0.8437, 0.8615,
         0.7831, 0.8594, 0.9007, 0.8541, 0.7352, 0.8081, 0.7834, 0.6526, 0.7157,
         0.8200]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:75
oneyr_surv_test: [0.7183817]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9576, 0.9351, 0.9134, 0.8965, 0.9397, 0.8675, 0.8137, 0.8170, 0.6627,
         0.7175, 0.7840, 0.8737, 0.8077, 0.6933, 0.8199, 0.7881, 0.6496, 0.6436,
         0.8297]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:76
oneyr_surv_test: [0.5976787]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9789, 0.9587, 0.9461, 0.9289, 0.9406, 0.8979, 0.8310, 0.8287, 0.8100,
         0.7744, 0.8377, 0.8983, 0.8340, 0.7122, 0.8196, 0.7932, 0.6508, 0.7045,
         0.8214]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:77
oneyr_surv_test: [0.6966459]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9573, 0.9152, 0.8710, 0.8857, 0.9348, 0.8483, 0.8169, 0.7910, 0.5882,
         0.6995, 0.7491, 0.8719, 0.8027, 0.6749, 0.8109, 0.7927, 0.6238, 0.6144,
         0.8389]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:78
oneyr_surv_test: [0.53600943]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9580, 0.9339, 0.9139, 0.8930, 0.9386, 0.8641, 0.8188, 0.8251, 0.6518,
         0.7197, 0.7837, 0.8693, 0.8070, 0.6984, 0.8223, 0.7892, 0.6551, 0.6401,
         0.8316]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:79
oneyr_surv_test: [0.5922661]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9562, 0.9326, 0.9098, 0.8865, 0.9360, 0.8583, 0.8151, 0.8214, 0.6357,
         0.7091, 0.7726, 0.8682, 0.8008, 0.6944, 0.8215, 0.7903, 0.6478, 0.6374,
         0.8295]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:80
oneyr_surv_test: [0.5777492]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9784, 0.9576, 0.9451, 0.9359, 0.9408, 0.8945, 0.8443, 0.8382, 0.8268,
         0.7666, 0.8459, 0.8933, 0.8440, 0.7250, 0.8113, 0.7855, 0.6535, 0.6974,
         0.8265]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:81
oneyr_surv_test: [0.6973545]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9552, 0.9308, 0.9097, 0.8881, 0.9383, 0.8595, 0.8188, 0.8286, 0.6260,
         0.7211, 0.7753, 0.8686, 0.8001, 0.7024, 0.8277, 0.7879, 0.6589, 0.6352,
         0.8304]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:82
oneyr_surv_test: [0.5792954]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9539, 0.9253, 0.8988, 0.8725, 0.9360, 0.8579, 0.8156, 0.8285, 0.5767,
         0.7172, 0.7530, 0.8627, 0.7901, 0.6999, 0.8304, 0.7905, 0.6577, 0.6246,
         0.8305]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:83
oneyr_surv_test: [0.5558243]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9707, 0.9473, 0.9268, 0.9155, 0.9404, 0.8901, 0.8162, 0.8085, 0.7484,
         0.7397, 0.8059, 0.8849, 0.8280, 0.6861, 0.8064, 0.7914, 0.6361, 0.6703,
         0.8340]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], device='cuda:1')
subj_num:84
oneyr_surv_test: [0.6530127]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9873, 0.9684, 0.9532, 0.9478, 0.9458, 0.9240, 0.8447, 0.8279, 0.9012,
         0.7894, 0.8596, 0.9232, 0.8758, 0.7174, 0.7816, 0.7856, 0.6152, 0.7429,
         0.8178]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:85
oneyr_surv_test: [0.7548069]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9738, 0.9524, 0.9431, 0.9221, 0.9398, 0.8832, 0.8398, 0.8469, 0.7665,
         0.7731, 0.8282, 0.8784, 0.8211, 0.7298, 0.8305, 0.7897, 0.6825, 0.6824,
         0.8230]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:86
oneyr_surv_test: [0.6694841]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9561, 0.9321, 0.9096, 0.8912, 0.9387, 0.8631, 0.8162, 0.8201, 0.6456,
         0.7134, 0.7800, 0.8711, 0.8063, 0.6948, 0.8206, 0.7884, 0.6505, 0.6375,
         0.8319]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:87
oneyr_surv_test: [0.5853865]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9810, 0.9611, 0.9364, 0.9390, 0.9411, 0.9084, 0.8245, 0.7856, 0.8647,
         0.7275, 0.8412, 0.9113, 0.8662, 0.6770, 0.7625, 0.7911, 0.5835, 0.7107,
         0.8317]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:88
oneyr_surv_test: [0.7086607]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9798, 0.9582, 0.9523, 0.9253, 0.9422, 0.9004, 0.8404, 0.8618, 0.7859,
         0.8087, 0.8323, 0.8899, 0.8215, 0.7494, 0.8505, 0.7893, 0.7085, 0.7071,
         0.8133]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:89
oneyr_surv_test: [0.70175177]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9737, 0.9528, 0.9426, 0.9188, 0.9410, 0.8896, 0.8310, 0.8422, 0.7675,
         0.7686, 0.8240, 0.8843, 0.8226, 0.7220, 0.8314, 0.7885, 0.6756, 0.6887,
         0.8220]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:90
oneyr_surv_test: [0.6726263]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9609, 0.9379, 0.9201, 0.8973, 0.9402, 0.8686, 0.8196, 0.8283, 0.6702,
         0.7261, 0.7897, 0.8711, 0.8089, 0.7016, 0.8247, 0.7886, 0.6581, 0.6485,
         0.8308]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:91
oneyr_surv_test: [0.6075453]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9871, 0.9667, 0.9570, 0.9439, 0.9449, 0.9241, 0.8482, 0.8499, 0.8856,
         0.8123, 0.8596, 0.9176, 0.8644, 0.7315, 0.8053, 0.7874, 0.6515, 0.7374,
         0.8167]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:92
oneyr_surv_test: [0.75263244]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9635, 0.9396, 0.9190, 0.9075, 0.9369, 0.8649, 0.8255, 0.8231, 0.6901,
         0.7292, 0.7932, 0.8729, 0.8115, 0.7045, 0.8197, 0.7915, 0.6505, 0.6492,
         0.8308]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:93
oneyr_surv_test: [0.6118445]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9607, 0.9378, 0.9199, 0.8967, 0.9400, 0.8682, 0.8196, 0.8281, 0.6669,
         0.7263, 0.7886, 0.8708, 0.8079, 0.7016, 0.8251, 0.7891, 0.6588, 0.6476,
         0.8302]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], device='cuda:1')
subj_num:94
oneyr_surv_test: [0.60652924]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9860, 0.9660, 0.9543, 0.9467, 0.9435, 0.9148, 0.8501, 0.8418, 0.8864,
         0.7918, 0.8645, 0.9087, 0.8657, 0.7333, 0.8022, 0.7841, 0.6440, 0.7332,
         0.8180]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:95
oneyr_surv_test: [0.7426944]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9740, 0.9491, 0.9402, 0.9050, 0.9404, 0.8879, 0.8363, 0.8622, 0.6942,
         0.7982, 0.7990, 0.8789, 0.7987, 0.7471, 0.8587, 0.7920, 0.7133, 0.6788,
         0.8186]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:96
oneyr_surv_test: [0.65673655]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9677, 0.9431, 0.9301, 0.9020, 0.9397, 0.8751, 0.8305, 0.8406, 0.6946,
         0.7559, 0.8043, 0.8727, 0.8109, 0.7158, 0.8321, 0.7913, 0.6756, 0.6597,
         0.8288]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:97
oneyr_surv_test: [0.6296318]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9804, 0.9582, 0.9518, 0.9236, 0.9412, 0.9039, 0.8350, 0.8580, 0.7734,
         0.8152, 0.8263, 0.8924, 0.8146, 0.7452, 0.8531, 0.7911, 0.7086, 0.7066,
         0.8134]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:98
oneyr_surv_test: [0.70259917]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9551, 0.9309, 0.9066, 0.8850, 0.9356, 0.8560, 0.8140, 0.8189, 0.6286,
         0.7051, 0.7692, 0.8683, 0.8003, 0.6922, 0.8206, 0.7904, 0.6450, 0.6345,
         0.8301]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:99
oneyr_surv_test: [0.5712896]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9819, 0.9608, 0.9551, 0.9278, 0.9412, 0.9038, 0.8442, 0.8653, 0.8104,
         0.8116, 0.8423, 0.8923, 0.8295, 0.7514, 0.8471, 0.7870, 0.7041, 0.7153,
         0.8128]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:100
oneyr_surv_test: [0.7111733]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9637, 0.9416, 0.9248, 0.8978, 0.9382, 0.8706, 0.8209, 0.8300, 0.6853,
         0.7310, 0.7943, 0.8719, 0.8085, 0.7043, 0.8259, 0.7898, 0.6572, 0.6553,
         0.8270]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], device='cuda:1')
subj_num:101
oneyr_surv_test: [0.6154282]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9813, 0.9597, 0.9530, 0.9265, 0.9415, 0.9070, 0.8373, 0.8568, 0.7911,
         0.8178, 0.8346, 0.8962, 0.8209, 0.7428, 0.8494, 0.7894, 0.7040, 0.7103,
         0.8136]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:102
oneyr_surv_test: [0.7100444]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9572, 0.9302, 0.9100, 0.8752, 0.9376, 0.8623, 0.8222, 0.8406, 0.5966,
         0.7310, 0.7645, 0.8626, 0.7921, 0.7110, 0.8368, 0.7901, 0.6706, 0.6343,
         0.8289]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:103
oneyr_surv_test: [0.5733334]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9868, 0.9658, 0.9506, 0.9422, 0.9417, 0.9171, 0.8411, 0.8162, 0.8732,
         0.7922, 0.8574, 0.9092, 0.8611, 0.7109, 0.7949, 0.7954, 0.6282, 0.7319,
         0.8231]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:104
oneyr_surv_test: [0.73719305]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9910, 0.9688, 0.9642, 0.9564, 0.9431, 0.9204, 0.8804, 0.9026, 0.9098,
         0.8703, 0.8546, 0.9353, 0.8628, 0.8080, 0.8377, 0.7766, 0.7199, 0.7541,
         0.7920]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device='cuda:1')
subj_num:105
oneyr_surv_test: [0.7685569]
y_pred.size:torch.Size([1, 19])
y_pred:tensor([[0.9630, 0.9367, 0.9216, 0.8903, 0.9390, 0.8685, 0.8312, 0.8476, 0.6333,
         0.7561, 0.7825, 0.8650, 0.7964, 0.7245, 0.8417, 0.7919, 0.6870, 0.6448,
         0.8273]], device='cuda:1', grad_fn=<SqueezeBackward1>)
labels.size:torch.Size([1, 38])
labels:tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:1')
subj_num:106
oneyr_surv_test: [0.60366845]
df_DL_score_test.shape:(107, 20)
107
duration_test.shape:(107,)
oneyr_survs_test.shape:(107,)
event_test.shape:(107,)
Original C-index for valid: 0.5379
95% CI for C-index for valid: (0.4591, 0.6167)
BS score at 365:[0.20072307]
seq range:min -19.66729164123535-max 1296.5587158203125
seq range:min -110.31443786621094-max 2376.626220703125
seq range:min -86.1859130859375-max 2738.498779296875
seq range:min 0.0-max 3345.15869140625
x.shape:torch.Size([1, 4, 224, 224, 224])
selected_seq:t1ce, 2