args:{'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'T_0': 10, 'lr': 0.001, 'min_lr': 1e-06, 'weight_decay': 1e-05, 'n_fold': 10, 'smoothing': 0.3, 'net_architect': 'SEResNext50', 'compart_name': 'resized', 'sequence': ['t1', 't2', 't1ce', 'flair'], 'data_key_list': ['sex', 'age', 'IDH', 'MGMT', 'GBL', 'EOR', 'duration_death', 'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion'], 'batch_size': 64, 'root_dir': '/mnt/hdd3/mskim/GBL', 'data_dir': '/mnt/hdd3/mskim/GBL/data', 'label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels', 'proc_label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels/proc_labels', 'exp_dir': '/mnt/hdd3/mskim/GBL/code/experiment', 'dataset_name': ['SNUH']}
Using SEResNext50
n_intervals: 19
Train dataset_name:SNUH_UCSF_UPenn_TCGA
Training on GPU 0
Setting seed:123456
dataset:SNUH for training
df_dataset.shape:(1134, 11)
dataset:UCSF for training
df_dataset.shape:(500, 11)
dataset:UPenn for training
df_dataset.shape:(425, 11)
dataset:TCGA for training
df_dataset.shape:(160, 11)
df_label_dataset_list.shape:(2219, 11)
saving new label csv file for SNUH_UCSF_UPenn_TCGA at /mnt/hdd3/mskim/GBL/data/label/surv_labels/SNUH_UCSF_UPenn_TCGA_OS_all_death.csv
dataset:severance for training
ext_df_dataset.shape:(132, 11)
saving new label csv file for external set severance at /mnt/hdd3/mskim/GBL/data/label/surv_labels/severance_OS_all_death.csv
Already copyied images of SNUH_UCSF_UPenn_TCGA for training to /mnt/hdd3/mskim/GBL/data/SNUH_UCSF_UPenn_TCGA/resized_BraTS path
df.index.values:2219
list_dataset: ['SNUH', 'UCSF', 'UPenn', 'TCGA']
split_dataset: SNUH
split_img_label_comm_list:1132
split_dataset: UCSF
split_img_label_comm_list:494
split_dataset: UPenn
split_img_label_comm_list:383
split_dataset: TCGA
split_img_label_comm_list:160
img_label_comm_list:2169
dataset_name:SNUH_UCSF_UPenn_TCGA, 2169
SNUH_UCSF_UPenn_TCGA df.shape: (2169, 11)
n_intervals: 19
df.index.values:132
img_label_comm_list:132
dataset_name:severance, 132
severance df.shape: (132, 11)
n_intervals: 19
train transform:
landmark_dataset_name:SNUH_UCSF_UPenn_TCGA
transform for SNUH_UCSF_UPenn_TCGA was obtained
valid transform:
landmark_dataset_name:SNUH_UCSF_UPenn_TCGA
transform for SNUH_UCSF_UPenn_TCGA was obtained
test transform:
landmark_dataset_name:severance
transform for severance was obtained
num of train_data: 1952
num of valid_data: 217
Epoch 1/200
----------
phase:train
train Loss: 2.6770
phase:valid
valid Loss: 2.4419

Epoch 2/200
----------
phase:train
train Loss: 2.3554
phase:valid
valid Loss: 2.4436

Epoch 3/200
----------
phase:train
train Loss: 2.3319
phase:valid
valid Loss: 2.3614

Epoch 4/200
----------
phase:train
train Loss: 2.2932
phase:valid
valid Loss: 2.3207

Epoch 5/200
----------
phase:train
train Loss: 2.2615
phase:valid
valid Loss: 2.3266

Epoch 6/200
----------
phase:train
train Loss: 2.2173
phase:valid
valid Loss: 2.2834

Epoch 7/200
----------
phase:train
train Loss: 2.1822
phase:valid
valid Loss: 2.2592

Epoch 8/200
----------
phase:train
train Loss: 2.1524
phase:valid
valid Loss: 2.2460

Epoch 9/200
----------
phase:train
train Loss: 2.1269
phase:valid
valid Loss: 2.2358

Epoch 10/200
----------
phase:train
train Loss: 2.1292
phase:valid
valid Loss: 2.2247

Epoch 11/200
----------
phase:train
train Loss: 2.1013
phase:valid
valid Loss: 2.2269

Epoch 12/200
----------
phase:train
train Loss: 2.1146
phase:valid
valid Loss: 2.2210

Epoch 13/200
----------
phase:train
train Loss: 2.1126
phase:valid
valid Loss: 2.2330

Epoch 14/200
----------
phase:train
train Loss: 2.1327
phase:valid
valid Loss: 2.2170

Epoch 15/200
----------
phase:train
train Loss: 2.1338
phase:valid
valid Loss: 2.2045

Epoch 16/200
----------
phase:train
train Loss: 2.1339
phase:valid
valid Loss: 2.2162

Epoch 17/200
----------
phase:train
train Loss: 2.1456
phase:valid
valid Loss: 2.3888

Epoch 18/200
----------
phase:train
train Loss: 2.1417
phase:valid
valid Loss: 2.2218

Epoch 19/200
----------
phase:train
train Loss: 2.1477
phase:valid
valid Loss: 2.2885

Epoch 20/200
----------
phase:train
train Loss: 2.1379
phase:valid
valid Loss: 2.2436

Epoch 21/200
----------
phase:train
train Loss: 2.1179
phase:valid
valid Loss: 2.2322

Epoch 22/200
----------
phase:train
train Loss: 2.1280
phase:valid
valid Loss: 2.3404

Epoch 23/200
----------
phase:train
train Loss: 2.1169
phase:valid
valid Loss: 2.1991

Epoch 24/200
----------
phase:train
train Loss: 2.0815
phase:valid
valid Loss: 2.2468

Epoch 25/200
----------
phase:train
train Loss: 2.0549
phase:valid
valid Loss: 2.2779

Epoch 26/200
----------
phase:train
train Loss: 2.0343
phase:valid
valid Loss: 2.1785

Epoch 27/200
----------
phase:train
train Loss: 2.0026
phase:valid
valid Loss: 2.2008

Epoch 28/200
----------
phase:train
train Loss: 1.9878
phase:valid
valid Loss: 2.1563

Epoch 29/200
----------
phase:train
train Loss: 1.9738
phase:valid
valid Loss: 2.1855

Epoch 30/200
----------
phase:train
train Loss: 1.9594
phase:valid
valid Loss: 2.1817

Epoch 31/200
----------
phase:train
train Loss: 1.9539
phase:valid
valid Loss: 2.1758

Epoch 32/200
----------
phase:train
train Loss: 1.9529
phase:valid
valid Loss: 2.1805

Epoch 33/200
----------
phase:train
train Loss: 1.9691
phase:valid
valid Loss: 2.1709

Epoch 34/200
----------
phase:train
train Loss: 1.9508
phase:valid
valid Loss: 2.1749

Epoch 35/200
----------
phase:train
train Loss: 1.9663
phase:valid
valid Loss: 2.1981

Epoch 36/200
----------
phase:train
train Loss: 1.9870
phase:valid
valid Loss: 2.3325

Epoch 37/200
----------
phase:train
train Loss: 2.0183
phase:valid
valid Loss: 2.4330

Epoch 38/200
----------
phase:train
train Loss: 2.0642
phase:valid
valid Loss: 2.2835

Epoch 39/200
----------
phase:train
train Loss: 2.0413
phase:valid
valid Loss: 2.2942

Epoch 40/200
----------
phase:train
train Loss: 2.0374
phase:valid
valid Loss: 2.1885

Epoch 41/200
----------
phase:train
train Loss: 2.0256
phase:valid
valid Loss: 2.3617

Epoch 42/200
----------
phase:train
train Loss: 2.0112
phase:valid
valid Loss: 2.2249

Epoch 43/200
----------
phase:train
train Loss: 2.0059
phase:valid
valid Loss: 2.3518

Epoch 44/200
----------
phase:train
train Loss: 1.9669
phase:valid
valid Loss: 2.2784

Epoch 45/200
----------
phase:train
train Loss: 1.9437
phase:valid
valid Loss: 2.2511

Epoch 46/200
----------
phase:train
train Loss: 1.9299
phase:valid
valid Loss: 2.3128

Epoch 47/200
----------
phase:train
train Loss: 1.8697
phase:valid
valid Loss: 2.1501

Epoch 48/200
----------
phase:train
train Loss: 1.8374
phase:valid
valid Loss: 2.2256

Epoch 49/200
----------
phase:train
train Loss: 1.7896
phase:valid
valid Loss: 2.2324

Epoch 50/200
----------
phase:train
train Loss: 1.7700
phase:valid
valid Loss: 2.2062

Epoch 51/200
----------
phase:train
train Loss: 1.7677
phase:valid
valid Loss: 2.2477

Epoch 52/200
----------
phase:train
train Loss: 1.7542
phase:valid
valid Loss: 2.2101

Epoch 53/200
----------
phase:train
train Loss: 1.7591
phase:valid
valid Loss: 2.2685

Epoch 54/200
----------
phase:train
train Loss: 1.7627
phase:valid
valid Loss: 2.2362

Epoch 55/200
----------
phase:train
train Loss: 1.7821
phase:valid
valid Loss: 2.3733

Epoch 56/200
----------
phase:train
train Loss: 1.8325
phase:valid
valid Loss: 2.3493

Epoch 57/200
----------
phase:train
train Loss: 1.8425
phase:valid
valid Loss: 2.4616

Epoch 58/200
----------
phase:train
train Loss: 1.8758
phase:valid
valid Loss: 2.2750

Epoch 59/200
----------
phase:train
train Loss: 1.8652
phase:valid
valid Loss: 2.3947

Epoch 60/200
----------
phase:train
train Loss: 1.9112
phase:valid
valid Loss: 2.2736

Epoch 61/200
----------
phase:train
train Loss: 1.8954
phase:valid
valid Loss: 2.5931

Epoch 62/200
----------
phase:train
train Loss: 1.8876
phase:valid
valid Loss: 2.1341

Epoch 63/200
----------
phase:train
train Loss: 1.8395
phase:valid
valid Loss: 2.1807

Epoch 64/200
----------
phase:train
train Loss: 1.7837
phase:valid
valid Loss: 2.1936

Epoch 65/200
----------
phase:train
train Loss: 1.7709
phase:valid
valid Loss: 2.2911

Epoch 66/200
----------
phase:train
train Loss: 1.6636
phase:valid
valid Loss: 2.2141

Epoch 67/200
----------
phase:train
train Loss: 1.6261
phase:valid
valid Loss: 2.4438

Epoch 68/200
----------
phase:train
train Loss: 1.5477
phase:valid
valid Loss: 2.2066

Epoch 69/200
----------
phase:train
train Loss: 1.4619
phase:valid
valid Loss: 2.1957

Epoch 70/200
----------
phase:train
train Loss: 1.4233
phase:valid
valid Loss: 2.2088

Epoch 71/200
----------
phase:train
train Loss: 1.4050
phase:valid
valid Loss: 2.2136

Epoch 72/200
----------
phase:train
train Loss: 1.4079
phase:valid
valid Loss: 2.2420

Epoch 73/200
----------
phase:train
train Loss: 1.4206
phase:valid
valid Loss: 2.2677

Epoch 74/200
----------
phase:train
train Loss: 1.4015
phase:valid
valid Loss: 2.2318

Epoch 75/200
----------
phase:train
train Loss: 1.4324
phase:valid
valid Loss: 2.4271

Epoch 76/200
----------
phase:train
train Loss: 1.5231
phase:valid
valid Loss: 2.6338

Epoch 77/200
----------
phase:train
train Loss: 1.5657
phase:valid
valid Loss: 2.2258

Epoch 78/200
----------
phase:train
train Loss: 1.6060
phase:valid
valid Loss: 2.2869

Epoch 79/200
----------
phase:train
train Loss: 1.6267
phase:valid
valid Loss: 2.2781

Epoch 80/200
----------
phase:train
train Loss: 1.6293
phase:valid
valid Loss: 2.4408

Epoch 81/200
----------
phase:train
train Loss: 1.6957
phase:valid
valid Loss: 2.3278

Epoch 82/200
----------
phase:train
train Loss: 1.6003
phase:valid
valid Loss: 2.3538

Epoch 83/200
----------
phase:train
train Loss: 1.5527
phase:valid
valid Loss: 2.2598

Epoch 84/200
----------
phase:train
train Loss: 1.4816
phase:valid
valid Loss: 2.8446

Epoch 85/200
----------
phase:train
train Loss: 1.4550
phase:valid
valid Loss: 2.3644

Epoch 86/200
----------
phase:train
train Loss: 1.2760
phase:valid
valid Loss: 2.4240

Epoch 87/200
----------
phase:train
train Loss: 1.1660
phase:valid
valid Loss: 2.3696

Epoch 88/200
----------
phase:train
train Loss: 1.1048
phase:valid
valid Loss: 2.4353

Epoch 89/200
----------
phase:train
train Loss: 0.9907
phase:valid
valid Loss: 2.4239

Epoch 90/200
----------
phase:train
train Loss: 0.9443
phase:valid
valid Loss: 2.4336

Epoch 91/200
----------
phase:train
train Loss: 0.9427
phase:valid
valid Loss: 2.4279

Epoch 92/200
----------
phase:train
train Loss: 0.9457
phase:valid
valid Loss: 2.4426

Epoch 93/200
----------
phase:train
train Loss: 0.9349
phase:valid
valid Loss: 2.5003

Epoch 94/200
----------
phase:train
train Loss: 0.9487
phase:valid
valid Loss: 2.5938

Epoch 95/200
----------
phase:train
train Loss: 1.0093
phase:valid
valid Loss: 2.5691

Epoch 96/200
----------
phase:train
train Loss: 1.0586
phase:valid
valid Loss: 2.3911

Epoch 97/200
----------
phase:train
train Loss: 1.1499
phase:valid
valid Loss: 2.5455

Epoch 98/200
----------
phase:train
train Loss: 1.2413
phase:valid
valid Loss: 2.5127

Epoch 99/200
----------
phase:train
train Loss: 1.2899
phase:valid
valid Loss: 2.3839

Epoch 100/200
----------
phase:train
train Loss: 1.3274
phase:valid
valid Loss: 2.3941

Epoch 101/200
----------
phase:train
train Loss: 1.3223
phase:valid
valid Loss: 2.4107

Epoch 102/200
----------
phase:train
train Loss: 1.2730
phase:valid
valid Loss: 2.4154

Epoch 103/200
----------
phase:train
train Loss: 1.2242
phase:valid
valid Loss: 2.4551

Epoch 104/200
----------
phase:train
train Loss: 1.1577
phase:valid
valid Loss: 2.5099

Epoch 105/200
----------
phase:train
train Loss: 1.0250
phase:valid
valid Loss: 2.5030

Epoch 106/200
----------
phase:train
train Loss: 0.9244
phase:valid
valid Loss: 2.6325

Epoch 107/200
----------
phase:train
train Loss: 0.8023
phase:valid
valid Loss: 2.6352

Epoch 108/200
----------
phase:train
train Loss: 0.7472
phase:valid
valid Loss: 2.7246

Epoch 109/200
----------
phase:train
train Loss: 0.6793
phase:valid
valid Loss: 2.8444

Epoch 110/200
----------
phase:train
train Loss: 0.6692
phase:valid
valid Loss: 2.7957

Epoch 111/200
----------
phase:train
train Loss: 0.6221
phase:valid
valid Loss: 2.8076

Epoch 112/200
----------
phase:train
train Loss: 0.6294
phase:valid
valid Loss: 2.8842

Epoch 113/200
----------
phase:train
train Loss: 0.6201
phase:valid
valid Loss: 2.8832

Epoch 114/200
----------
phase:train
train Loss: 0.6298
phase:valid
valid Loss: 2.8775

Epoch 115/200
----------
phase:train
train Loss: 0.6416
phase:valid
valid Loss: 2.8736

Epoch 116/200
----------
phase:train
train Loss: 0.7231
phase:valid
valid Loss: 2.9822

Epoch 117/200
----------
phase:train
train Loss: 0.8182
phase:valid
valid Loss: 2.8340

Epoch 118/200
----------
phase:train
train Loss: 0.8926
phase:valid
valid Loss: 2.8445

Epoch 119/200
----------
phase:train
train Loss: 1.0562
phase:valid
valid Loss: 2.4981

Epoch 120/200
----------
phase:train
train Loss: 1.1045
phase:valid
valid Loss: 2.5581

Epoch 121/200
----------
phase:train
train Loss: 1.0055
phase:valid
valid Loss: 2.8425

Epoch 122/200
----------
phase:train
train Loss: 0.9839
phase:valid
valid Loss: 2.6759

Epoch 123/200
----------
phase:train
train Loss: 0.8997
phase:valid
valid Loss: 2.8584

Epoch 124/200
----------
phase:train
train Loss: 0.8930
phase:valid
valid Loss: 2.8481

Epoch 125/200
----------
phase:train
train Loss: 0.8000
phase:valid
valid Loss: 2.7357

Epoch 126/200
----------
phase:train
train Loss: 0.6740
phase:valid
valid Loss: 2.6420

Epoch 127/200
----------
phase:train
train Loss: 0.6177
phase:valid
valid Loss: 2.9479

Epoch 128/200
----------
phase:train
train Loss: 0.5561
phase:valid
valid Loss: 2.9763

Epoch 129/200
----------
phase:train
train Loss: 0.5075
phase:valid
valid Loss: 3.1382

Epoch 130/200
----------
phase:train
train Loss: 0.5148
phase:valid
valid Loss: 3.1646

Epoch 131/200
----------
phase:train
train Loss: 0.4938
phase:valid
valid Loss: 3.1513

Epoch 132/200
----------
phase:train
train Loss: 0.5001
phase:valid
valid Loss: 3.1646

Epoch 133/200
----------
phase:train
train Loss: 0.4932
phase:valid
valid Loss: 3.1147

Epoch 134/200
----------
phase:train
train Loss: 0.5059
phase:valid
valid Loss: 3.2091

Epoch 135/200
----------
phase:train
train Loss: 0.5050
phase:valid
valid Loss: 3.3866

Epoch 136/200
----------
phase:train
train Loss: 0.5476
phase:valid
valid Loss: 3.1123

Epoch 137/200
----------
phase:train
train Loss: 0.5994
phase:valid
valid Loss: 3.2488

Epoch 138/200
----------
phase:train
train Loss: 0.7753
phase:valid
valid Loss: 3.1368

Epoch 139/200
----------
phase:train
train Loss: 0.8063
phase:valid
valid Loss: 3.2094

Epoch 140/200
----------
phase:train
train Loss: 0.8629
phase:valid
valid Loss: 2.7726

Epoch 141/200
----------
phase:train
train Loss: 0.8996
phase:valid
valid Loss: 2.6971

Epoch 142/200
----------
phase:train
train Loss: 0.8495
phase:valid
valid Loss: 2.6547

Epoch 143/200
----------
phase:train
train Loss: 0.7847
phase:valid
valid Loss: 2.6357

Epoch 144/200
----------
phase:train
train Loss: 0.7347
phase:valid
valid Loss: 3.2508

Epoch 145/200
----------
phase:train
train Loss: 0.6703
phase:valid
valid Loss: 3.1296

Epoch 146/200
----------
phase:train
train Loss: 0.5901
phase:valid
valid Loss: 3.1129

Epoch 147/200
----------
phase:train
train Loss: 0.5402
phase:valid
valid Loss: 3.1465

Epoch 148/200
----------
phase:train
train Loss: 0.4863
phase:valid
valid Loss: 3.2440

Epoch 149/200
----------
phase:train
train Loss: 0.4875
phase:valid
valid Loss: 3.2956

Epoch 150/200
----------
phase:train
train Loss: 0.4904
phase:valid
valid Loss: 3.3808

Epoch 151/200
----------
phase:train
train Loss: 0.4737
phase:valid
valid Loss: 3.2919

Epoch 152/200
----------
phase:train
train Loss: 0.4599
phase:valid
valid Loss: 3.4092

Epoch 153/200
----------
phase:train
train Loss: 0.4423
phase:valid
valid Loss: 3.4158

Epoch 154/200
----------
phase:train
train Loss: 0.4512
phase:valid
valid Loss: 3.4607

Epoch 155/200
----------
phase:train
train Loss: 0.4570
phase:valid
valid Loss: 3.4427

Epoch 156/200
----------
phase:train
train Loss: 0.5060
phase:valid
valid Loss: 3.6857

Epoch 157/200
----------
phase:train
train Loss: 0.5702
phase:valid
valid Loss: 3.2153

Epoch 158/200
----------
phase:train
train Loss: 0.6399
phase:valid
valid Loss: 3.8446

Epoch 159/200
----------
phase:train
train Loss: 0.7473
phase:valid
valid Loss: 3.2852

Epoch 160/200
----------
phase:train
train Loss: 0.7752
phase:valid
valid Loss: 3.6366

Epoch 161/200
----------
phase:train
train Loss: 0.8439
phase:valid
valid Loss: 4.1608

Epoch 162/200
----------
phase:train
train Loss: 0.7724
phase:valid
valid Loss: 3.3340

Epoch 163/200
----------
phase:train
train Loss: 0.7658
phase:valid
valid Loss: 3.0956

Epoch 164/200
----------
phase:train
train Loss: 0.6705
phase:valid
valid Loss: 2.9810

Epoch 165/200
----------
phase:train
train Loss: 0.6393
phase:valid
valid Loss: 3.1769

Epoch 166/200
----------
phase:train
train Loss: 0.5761
phase:valid
valid Loss: 3.5361

Epoch 167/200
----------
phase:train
train Loss: 0.5146
phase:valid
valid Loss: 3.1707

Epoch 168/200
----------
phase:train
train Loss: 0.4629
phase:valid
valid Loss: 3.4407

Epoch 169/200
----------
phase:train
train Loss: 0.4794
phase:valid
valid Loss: 3.5099

Epoch 170/200
----------
phase:train
train Loss: 0.4745
phase:valid
valid Loss: 3.4255

Epoch 171/200
----------
phase:train
train Loss: 0.4796
phase:valid
valid Loss: 3.4832

Epoch 172/200
----------
phase:train
train Loss: 0.4747
phase:valid
valid Loss: 3.4671

Epoch 173/200
----------
phase:train
train Loss: 0.4718
phase:valid
valid Loss: 3.4829

Epoch 174/200
----------
phase:train
train Loss: 0.4043
phase:valid
valid Loss: 3.5879

Epoch 175/200
----------
phase:train
train Loss: 0.4668
phase:valid
valid Loss: 3.7339

Epoch 176/200
----------
phase:train
train Loss: 0.5716
phase:valid
valid Loss: 3.6491

Epoch 177/200
----------
phase:train
train Loss: 0.5906
phase:valid
valid Loss: 3.5077

Epoch 178/200
----------
phase:train
train Loss: 0.6596
phase:valid
valid Loss: 3.8597

Epoch 179/200
----------
phase:train
train Loss: 0.6866
phase:valid
valid Loss: 3.3032

Epoch 180/200
----------
phase:train
train Loss: 0.7595
phase:valid
valid Loss: 3.1625

Epoch 181/200
----------
phase:train
train Loss: 0.7946
phase:valid
valid Loss: 3.5236

Epoch 182/200
----------
phase:train
train Loss: 0.8378
phase:valid
valid Loss: 3.2461

Epoch 183/200
----------
phase:train
train Loss: 0.7290
phase:valid
valid Loss: 3.1879

Epoch 184/200
----------
phase:train
train Loss: 0.6443
phase:valid
valid Loss: 3.5672

Epoch 185/200
----------
phase:train
train Loss: 0.6576
phase:valid
valid Loss: 3.5151

Epoch 186/200
----------
phase:train
train Loss: 0.5942
phase:valid
valid Loss: 3.2840

Epoch 187/200
----------
phase:train
train Loss: 0.5738
phase:valid
valid Loss: 3.8699

Epoch 188/200
----------
phase:train
train Loss: 0.5418
phase:valid
valid Loss: 3.7537

Epoch 189/200
----------
phase:train
train Loss: 0.5204
phase:valid
valid Loss: 3.7378

Epoch 190/200
----------
phase:train
train Loss: 0.5368
phase:valid
valid Loss: 3.6831

Epoch 191/200
----------
phase:train
train Loss: 0.4854
phase:valid
valid Loss: 3.7969

Epoch 192/200
----------
phase:train
train Loss: 0.5081
phase:valid
valid Loss: 3.8707

Epoch 193/200
----------
phase:train
train Loss: 0.5238
phase:valid
valid Loss: 3.8649

Epoch 194/200
----------
phase:train
train Loss: 0.5454
phase:valid
valid Loss: 3.7638

Epoch 195/200
----------
phase:train
train Loss: 0.5147
phase:valid
valid Loss: 3.6964

Epoch 196/200
----------
phase:train
train Loss: 0.5299
phase:valid
valid Loss: 3.6772

Epoch 197/200
----------
phase:train
train Loss: 0.6431
phase:valid
valid Loss: 4.5232

Epoch 198/200
----------
phase:train
train Loss: 0.6447
phase:valid
valid Loss: 4.8388

Epoch 199/200
----------
phase:train
train Loss: 0.7581
phase:valid
valid Loss: 3.5914

Epoch 200/200
----------
phase:train
train Loss: 0.7986
phase:valid
valid Loss: 3.3935

Training complete in 5h 24m 28s
Best Loss  2.134119538118213