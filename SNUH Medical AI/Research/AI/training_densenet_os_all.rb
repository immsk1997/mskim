args:{'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'T_0': 10, 'lr': 0.001, 'min_lr': 1e-06, 'weight_decay': 1e-05, 'n_fold': 10, 'smoothing': 0.3, 'net_architect': 'DenseNet', 'compart_name': 'resized', 'sequence': ['t1', 't2', 't1ce', 'flair'], 'data_key_list': ['sex', 'age', 'IDH', 'MGMT', 'GBL', 'EOR', 'duration_death', 'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion'], 'batch_size': 64, 'root_dir': '/mnt/hdd3/mskim/GBL', 'data_dir': '/mnt/hdd3/mskim/GBL/data', 'label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels', 'proc_label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels/proc_labels', 'exp_dir': '/mnt/hdd3/mskim/GBL/code/experiment', 'dataset_name': ['SNUH']}
Using DenseNet
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
self.layer_name_list:['features']
num of train_data: 1952
num of valid_data: 217
Epoch 1/200
----------
phase:train
train Loss: 2.8244
phase:valid
valid Loss: 2.3447

Epoch 2/200
----------
phase:train
train Loss: 2.2254
phase:valid
valid Loss: 2.7037

Epoch 3/200
----------
phase:train
train Loss: 2.1828
phase:valid
valid Loss: 2.2583

Epoch 4/200
----------
phase:train
train Loss: 2.1762
phase:valid
valid Loss: 2.3270

Epoch 5/200
----------
phase:train
train Loss: 2.1349
phase:valid
valid Loss: 2.2879

Epoch 6/200
----------
phase:train
train Loss: 2.1130
phase:valid
valid Loss: 2.2487

Epoch 7/200
----------
phase:train
train Loss: 2.1075
phase:valid
valid Loss: 2.2224

Epoch 8/200
----------
phase:train
train Loss: 2.0902
phase:valid
valid Loss: 2.1740

Epoch 9/200
----------
phase:train
train Loss: 2.0556
phase:valid
valid Loss: 2.1647

Epoch 10/200
----------
phase:train
train Loss: 2.0450
phase:valid
valid Loss: 2.1730

Epoch 11/200
----------
phase:train
train Loss: 2.0429
phase:valid
valid Loss: 2.1678

Epoch 12/200
----------
phase:train
train Loss: 2.0449
phase:valid
valid Loss: 2.1702

Epoch 13/200
----------
phase:train
train Loss: 2.0507
phase:valid
valid Loss: 2.1718

Epoch 14/200
----------
phase:train
train Loss: 2.0481
phase:valid
valid Loss: 2.1638

Epoch 15/200
----------
phase:train
train Loss: 2.0675
phase:valid
valid Loss: 2.1625

Epoch 16/200
----------
phase:train
train Loss: 2.0830
phase:valid
valid Loss: 2.1767

Epoch 17/200
----------
phase:train
train Loss: 2.0834
phase:valid
valid Loss: 2.2490

Epoch 18/200
----------
phase:train
train Loss: 2.1002
phase:valid
valid Loss: 2.5625

Epoch 19/200
----------
phase:train
train Loss: 2.0998
phase:valid
valid Loss: 2.6280

Epoch 20/200
----------
phase:train
train Loss: 2.0999
phase:valid
valid Loss: 2.5047

Epoch 21/200
----------
phase:train
train Loss: 2.1005
phase:valid
valid Loss: 2.2620

Epoch 22/200
----------
phase:train
train Loss: 2.0645
phase:valid
valid Loss: 2.1861

Epoch 23/200
----------
phase:train
train Loss: 2.0902
phase:valid
valid Loss: 2.7949

Epoch 24/200
----------
phase:train
train Loss: 2.0642
phase:valid
valid Loss: 3.4308

Epoch 25/200
----------
phase:train
train Loss: 2.0554
phase:valid
valid Loss: 2.1802

Epoch 26/200
----------
phase:train
train Loss: 2.0142
phase:valid
valid Loss: 2.1779

Epoch 27/200
----------
phase:train
train Loss: 1.9793
phase:valid
valid Loss: 2.1493

Epoch 28/200
----------
phase:train
train Loss: 1.9632
phase:valid
valid Loss: 2.1432

Epoch 29/200
----------
phase:train
train Loss: 1.9395
phase:valid
valid Loss: 2.1317

Epoch 30/200
----------
phase:train
train Loss: 1.9175
phase:valid
valid Loss: 2.1671

Epoch 31/200
----------
phase:train
train Loss: 1.9224
phase:valid
valid Loss: 2.1523

Epoch 32/200
----------
phase:train
train Loss: 1.9188
phase:valid
valid Loss: 2.1603

Epoch 33/200
----------
phase:train
train Loss: 1.9170
phase:valid
valid Loss: 2.1366

Epoch 34/200
----------
phase:train
train Loss: 1.9317
phase:valid
valid Loss: 2.1085

Epoch 35/200
----------
phase:train
train Loss: 1.9481
phase:valid
valid Loss: 2.1714

Epoch 36/200
----------
phase:train
train Loss: 1.9719
phase:valid
valid Loss: 2.1990

Epoch 37/200
----------
phase:train
train Loss: 1.9703
phase:valid
valid Loss: 2.4075

Epoch 38/200
----------
phase:train
train Loss: 1.9802
phase:valid
valid Loss: 2.2154

Epoch 39/200
----------
phase:train
train Loss: 2.0088
phase:valid
valid Loss: 2.3078

Epoch 40/200
----------
phase:train
train Loss: 1.9990
phase:valid
valid Loss: 2.1999

Epoch 41/200
----------
phase:train
train Loss: 2.0060
phase:valid
valid Loss: 2.3693

Epoch 42/200
----------
phase:train
train Loss: 2.0126
phase:valid
valid Loss: 2.3601

Epoch 43/200
----------
phase:train
train Loss: 2.0006
phase:valid
valid Loss: 2.2206

Epoch 44/200
----------
phase:train
train Loss: 1.9344
phase:valid
valid Loss: 3.0540

Epoch 45/200
----------
phase:train
train Loss: 1.9147
phase:valid
valid Loss: 2.1118

Epoch 46/200
----------
phase:train
train Loss: 1.8826
phase:valid
valid Loss: 2.2040

Epoch 47/200
----------
phase:train
train Loss: 1.8483
phase:valid
valid Loss: 2.6001

Epoch 48/200
----------
phase:train
train Loss: 1.8126
phase:valid
valid Loss: 2.1844

Epoch 49/200
----------
phase:train
train Loss: 1.7675
phase:valid
valid Loss: 2.2259

Epoch 50/200
----------
phase:train
train Loss: 1.7492
phase:valid
valid Loss: 2.2128

Epoch 51/200
----------
phase:train
train Loss: 1.7414
phase:valid
valid Loss: 2.2198

Epoch 52/200
----------
phase:train
train Loss: 1.7321
phase:valid
valid Loss: 2.1915

Epoch 53/200
----------
phase:train
train Loss: 1.7302
phase:valid
valid Loss: 2.2024

Epoch 54/200
----------
phase:train
train Loss: 1.7544
phase:valid
valid Loss: 2.2918

Epoch 55/200
----------
phase:train
train Loss: 1.7628
phase:valid
valid Loss: 2.4381

Epoch 56/200
----------
phase:train
train Loss: 1.7941
phase:valid
valid Loss: 2.4484

Epoch 57/200
----------
phase:train
train Loss: 1.8429
phase:valid
valid Loss: 2.3726

Epoch 58/200
----------
phase:train
train Loss: 1.8430
phase:valid
valid Loss: 2.3131

Epoch 59/200
----------
phase:train
train Loss: 1.8878
phase:valid
valid Loss: 2.2725

Epoch 60/200
----------
phase:train
train Loss: 1.8719
phase:valid
valid Loss: 2.3113

Epoch 61/200
----------
phase:train
train Loss: 1.8553
phase:valid
valid Loss: 2.3885

Epoch 62/200
----------
phase:train
train Loss: 1.8480
phase:valid
valid Loss: 2.2498

Epoch 63/200
----------
phase:train
train Loss: 1.8313
phase:valid
valid Loss: 2.3870

Epoch 64/200
----------
phase:train
train Loss: 1.8066
phase:valid
valid Loss: 2.3194

Epoch 65/200
----------
phase:train
train Loss: 1.7584
phase:valid
valid Loss: 2.5439

Epoch 66/200
----------
phase:train
train Loss: 1.6787
phase:valid
valid Loss: 2.3409

Epoch 67/200
----------
phase:train
train Loss: 1.6069
phase:valid
valid Loss: 2.2831

Epoch 68/200
----------
phase:train
train Loss: 1.5321
phase:valid
valid Loss: 2.4727

Epoch 69/200
----------
phase:train
train Loss: 1.4797
phase:valid
valid Loss: 2.3515

Epoch 70/200
----------
phase:train
train Loss: 1.4141
phase:valid
valid Loss: 2.3541

Epoch 71/200
----------
phase:train
train Loss: 1.4112
phase:valid
valid Loss: 2.3228

Epoch 72/200
----------
phase:train
train Loss: 1.4080
phase:valid
valid Loss: 2.3250

Epoch 73/200
----------
phase:train
train Loss: 1.4112
phase:valid
valid Loss: 2.3362

Epoch 74/200
----------
phase:train
train Loss: 1.4230
phase:valid
valid Loss: 2.3780

Epoch 75/200
----------
phase:train
train Loss: 1.4536
phase:valid
valid Loss: 2.3593

Epoch 76/200
----------
phase:train
train Loss: 1.5317
phase:valid
valid Loss: 2.4240

Epoch 77/200
----------
phase:train
train Loss: 1.6149
phase:valid
valid Loss: 2.7731

Epoch 78/200
----------
phase:train
train Loss: 1.6194
phase:valid
valid Loss: 2.4492

Epoch 79/200
----------
phase:train
train Loss: 1.6522
phase:valid
valid Loss: 3.9144

Epoch 80/200
----------
phase:train
train Loss: 1.6738
phase:valid
valid Loss: 2.9323

Epoch 81/200
----------
phase:train
train Loss: 1.6437
phase:valid
valid Loss: 2.7395

Epoch 82/200
----------
phase:train
train Loss: 1.6339
phase:valid
valid Loss: 2.6558

Epoch 83/200
----------
phase:train
train Loss: 1.5853
phase:valid
valid Loss: 2.6610

Epoch 84/200
----------
phase:train
train Loss: 1.4887
phase:valid
valid Loss: 2.5713

Epoch 85/200
----------
phase:train
train Loss: 1.4356
phase:valid
valid Loss: 2.4240

Epoch 86/200
----------
phase:train
train Loss: 1.3513
phase:valid
valid Loss: 2.5318

Epoch 87/200
----------
phase:train
train Loss: 1.2519
phase:valid
valid Loss: 2.4932

Epoch 88/200
----------
phase:train
train Loss: 1.1541
phase:valid
valid Loss: 2.5968

Epoch 89/200
----------
phase:train
train Loss: 1.0765
phase:valid
valid Loss: 2.5681

Epoch 90/200
----------
phase:train
train Loss: 1.0189
phase:valid
valid Loss: 2.5756

Epoch 91/200
----------
phase:train
train Loss: 1.0131
phase:valid
valid Loss: 2.5508

Epoch 92/200
----------
phase:train
train Loss: 1.0127
phase:valid
valid Loss: 2.5629

Epoch 93/200
----------
phase:train
train Loss: 1.0278
phase:valid
valid Loss: 2.6442

Epoch 94/200
----------
phase:train
train Loss: 1.0352
phase:valid
valid Loss: 2.6468

Epoch 95/200
----------
phase:train
train Loss: 1.0649
phase:valid
valid Loss: 2.9557

Epoch 96/200
----------
phase:train
train Loss: 1.1997
phase:valid
valid Loss: 2.8495

Epoch 97/200
----------
phase:train
train Loss: 1.2663
phase:valid
valid Loss: 2.6995

Epoch 98/200
----------
phase:train
train Loss: 1.3453
phase:valid
valid Loss: 2.5621

Epoch 99/200
----------
phase:train
train Loss: 1.3690
phase:valid
valid Loss: 2.7211

Epoch 100/200
----------
phase:train
train Loss: 1.4142
phase:valid
valid Loss: 2.5840

Epoch 101/200
----------
phase:train
train Loss: 1.4024
phase:valid
valid Loss: 2.7716

Epoch 102/200
----------
phase:train
train Loss: 1.3893
phase:valid
valid Loss: 2.6222

Epoch 103/200
----------
phase:train
train Loss: 1.3184
phase:valid
valid Loss: 2.7402

Epoch 104/200
----------
phase:train
train Loss: 1.2241
phase:valid
valid Loss: 2.9523

Epoch 105/200
----------
phase:train
train Loss: 1.1686
phase:valid
valid Loss: 2.7838

Epoch 106/200
----------
phase:train
train Loss: 1.0315
phase:valid
valid Loss: 2.8525

Epoch 107/200
----------
phase:train
train Loss: 0.9237
phase:valid
valid Loss: 2.7607

Epoch 108/200
----------
phase:train
train Loss: 0.8437
phase:valid
valid Loss: 2.8215

Epoch 109/200
----------
phase:train
train Loss: 0.7832
phase:valid
valid Loss: 2.7838

Epoch 110/200
----------
phase:train
train Loss: 0.7331
phase:valid
valid Loss: 2.8558

Epoch 111/200
----------
phase:train
train Loss: 0.7290
phase:valid
valid Loss: 2.8045

Epoch 112/200
----------
phase:train
train Loss: 0.7198
phase:valid
valid Loss: 2.8213

Epoch 113/200
----------
phase:train
train Loss: 0.7261
phase:valid
valid Loss: 2.9001

Epoch 114/200
----------
phase:train
train Loss: 0.7213
phase:valid
valid Loss: 2.9455

Epoch 115/200
----------
phase:train
train Loss: 0.7408
phase:valid
valid Loss: 3.1036

Epoch 116/200
----------
phase:train
train Loss: 0.8368
phase:valid
valid Loss: 3.0070

Epoch 117/200
----------
phase:train
train Loss: 1.0153
phase:valid
valid Loss: 3.0196

Epoch 118/200
----------
phase:train
train Loss: 1.1189
phase:valid
valid Loss: 3.1801

Epoch 119/200
----------
phase:train
train Loss: 1.1890
phase:valid
valid Loss: 2.9421

Epoch 120/200
----------
phase:train
train Loss: 1.1828
phase:valid
valid Loss: 3.1625

Epoch 121/200
----------
phase:train
train Loss: 1.1768
phase:valid
valid Loss: 3.3873

Epoch 122/200
----------
phase:train
train Loss: 1.1304
phase:valid
valid Loss: 2.6432

Epoch 123/200
----------
phase:train
train Loss: 1.1010
phase:valid
valid Loss: 3.2698

Epoch 124/200
----------
phase:train
train Loss: 0.9776
phase:valid
valid Loss: 2.9966

Epoch 125/200
----------
phase:train
train Loss: 0.9019
phase:valid
valid Loss: 3.1854

Epoch 126/200
----------
phase:train
train Loss: 0.7806
phase:valid
valid Loss: 2.9796

Epoch 127/200
----------
phase:train
train Loss: 0.7039
phase:valid
valid Loss: 2.9562

Epoch 128/200
----------
phase:train
train Loss: 0.6301
phase:valid
valid Loss: 3.0300

Epoch 129/200
----------
phase:train
train Loss: 0.5915
phase:valid
valid Loss: 3.0295

Epoch 130/200
----------
phase:train
train Loss: 0.5730
phase:valid
valid Loss: 3.0058

Epoch 131/200
----------
phase:train
train Loss: 0.5563
phase:valid
valid Loss: 3.0283

Epoch 132/200
----------
phase:train
train Loss: 0.5554
phase:valid
valid Loss: 3.0618

Epoch 133/200
----------
phase:train
train Loss: 0.5485
phase:valid
valid Loss: 3.0197

Epoch 134/200
----------
phase:train
train Loss: 0.5527
phase:valid
valid Loss: 3.0965

Epoch 135/200
----------
phase:train
train Loss: 0.5943
phase:valid
valid Loss: 3.2178

Epoch 136/200
----------
phase:train
train Loss: 0.6437
phase:valid
valid Loss: 3.8435

Epoch 137/200
----------
phase:train
train Loss: 0.7795
phase:valid
valid Loss: 4.0769

Epoch 138/200
----------
phase:train
train Loss: 0.9077
phase:valid
valid Loss: 3.0420

Epoch 139/200
----------
phase:train
train Loss: 0.9870
phase:valid
valid Loss: 3.3594

Epoch 140/200
----------
phase:train
train Loss: 1.0964
phase:valid
valid Loss: 4.1801

Epoch 141/200
----------
phase:train
train Loss: 1.0645
phase:valid
valid Loss: 2.9548

Epoch 142/200
----------
phase:train
train Loss: 0.9440
phase:valid
valid Loss: 3.0450

Epoch 143/200
----------
phase:train
train Loss: 0.9535
phase:valid
valid Loss: 2.9880

Epoch 144/200
----------
phase:train
train Loss: 0.8538
phase:valid
valid Loss: 3.2623

Epoch 145/200
----------
phase:train
train Loss: 0.7718
phase:valid
valid Loss: 3.1107

Epoch 146/200
----------
phase:train
train Loss: 0.6750
phase:valid
valid Loss: 3.3339

Epoch 147/200
----------
phase:train
train Loss: 0.6003
phase:valid
valid Loss: 3.0281

Epoch 148/200
----------
phase:train
train Loss: 0.5448
phase:valid
valid Loss: 3.1042

Epoch 149/200
----------
phase:train
train Loss: 0.5058
phase:valid
valid Loss: 3.1089

Epoch 150/200
----------
phase:train
train Loss: 0.4890
phase:valid
valid Loss: 3.1714

Epoch 151/200
----------
phase:train
train Loss: 0.4982
phase:valid
valid Loss: 3.1224

Epoch 152/200
----------
phase:train
train Loss: 0.4942
phase:valid
valid Loss: 3.1559

Epoch 153/200
----------
phase:train
train Loss: 0.4918
phase:valid
valid Loss: 3.1838

Epoch 154/200
----------
phase:train
train Loss: 0.4805
phase:valid
valid Loss: 3.1592

Epoch 155/200
----------
phase:train
train Loss: 0.4817
phase:valid
valid Loss: 3.5139

Epoch 156/200
----------
phase:train
train Loss: 0.5442
phase:valid
valid Loss: 3.7480

Epoch 157/200
----------
phase:train
train Loss: 0.6290
phase:valid
valid Loss: 4.7819

Epoch 158/200
----------
phase:train
train Loss: 0.7860
phase:valid
valid Loss: 3.9860

Epoch 159/200
----------
phase:train
train Loss: 0.8833
phase:valid
valid Loss: 3.0650

Epoch 160/200
----------
phase:train
train Loss: 0.9013
phase:valid
valid Loss: 3.2169

Epoch 161/200
----------
phase:train
train Loss: 0.9148
phase:valid
valid Loss: 3.2568

Epoch 162/200
----------
phase:train
train Loss: 0.8895
phase:valid
valid Loss: 3.0680

Epoch 163/200
----------
phase:train
train Loss: 0.8195
phase:valid
valid Loss: 3.3621

Epoch 164/200
----------
phase:train
train Loss: 0.7568
phase:valid
valid Loss: 3.2923

Epoch 165/200
----------
phase:train
train Loss: 0.6913
phase:valid
valid Loss: 3.3867

Epoch 166/200
----------
phase:train
train Loss: 0.5718
phase:valid
valid Loss: 3.5880

Epoch 167/200
----------
phase:train
train Loss: 0.5234
phase:valid
valid Loss: 3.2747

Epoch 168/200
----------
phase:train
train Loss: 0.4844
phase:valid
valid Loss: 3.3413

Epoch 169/200
----------
phase:train
train Loss: 0.4598
phase:valid
valid Loss: 3.3722

Epoch 170/200
----------
phase:train
train Loss: 0.4415
phase:valid
valid Loss: 3.3949

Epoch 171/200
----------
phase:train
train Loss: 0.4386
phase:valid
valid Loss: 3.3841

Epoch 172/200
----------
phase:train
train Loss: 0.4295
phase:valid
valid Loss: 3.3999

Epoch 173/200
----------
phase:train
train Loss: 0.4341
phase:valid
valid Loss: 3.4235

Epoch 174/200
----------
phase:train
train Loss: 0.4340
phase:valid
valid Loss: 3.3690

Epoch 175/200
----------
phase:train
train Loss: 0.4481
phase:valid
valid Loss: 3.2576

Epoch 176/200
----------
phase:train
train Loss: 0.4681
phase:valid
valid Loss: 3.2784

Epoch 177/200
----------
phase:train
train Loss: 0.5266
phase:valid
valid Loss: 3.2563

Epoch 178/200
----------
phase:train
train Loss: 0.6317
phase:valid
valid Loss: 3.4023

Epoch 179/200
----------
phase:train
train Loss: 0.7646
phase:valid
valid Loss: 5.4257

Epoch 180/200
----------
phase:train
train Loss: 0.9174
phase:valid
valid Loss: 3.3504

Epoch 181/200
----------
phase:train
train Loss: 1.0059
phase:valid
valid Loss: 3.5016

Epoch 182/200
----------
phase:train
train Loss: 0.8673
phase:valid
valid Loss: 3.0628

Epoch 183/200
----------
phase:train
train Loss: 0.7573
phase:valid
valid Loss: 3.1788

Epoch 184/200
----------
phase:train
train Loss: 0.6894
phase:valid
valid Loss: 3.7307

Epoch 185/200
----------
phase:train
train Loss: 0.6063
phase:valid
valid Loss: 3.1703

Epoch 186/200
----------
phase:train
train Loss: 0.5451
phase:valid
valid Loss: 3.7409

Epoch 187/200
----------
phase:train
train Loss: 0.4929
phase:valid
valid Loss: 3.2408

Epoch 188/200
----------
phase:train
train Loss: 0.4493
phase:valid
valid Loss: 3.4721

Epoch 189/200
----------
phase:train
train Loss: 0.4157
phase:valid
valid Loss: 3.4577

Epoch 190/200
----------
phase:train
train Loss: 0.4246
phase:valid
valid Loss: 3.4815

Epoch 191/200
----------
phase:train
train Loss: 0.4044
phase:valid
valid Loss: 3.4092

Epoch 192/200
----------
phase:train
train Loss: 0.4025
phase:valid
valid Loss: 3.4237

Epoch 193/200
----------
phase:train
train Loss: 0.4022
phase:valid
valid Loss: 3.4350

Epoch 194/200
----------
phase:train
train Loss: 0.4124
phase:valid
valid Loss: 3.3772

Epoch 195/200
----------
phase:train
train Loss: 0.4085
phase:valid
valid Loss: 3.5365

Epoch 196/200
----------
phase:train
train Loss: 0.4168
phase:valid
valid Loss: 3.8571

Epoch 197/200
----------
phase:train
train Loss: 0.4974
phase:valid
valid Loss: 3.8167

Epoch 198/200
----------
phase:train
train Loss: 0.5999
phase:valid
valid Loss: 3.2045

Epoch 199/200
----------
phase:train
train Loss: 0.7129
phase:valid
valid Loss: 5.0882

Epoch 200/200
----------
phase:train
train Loss: 0.8100
phase:valid
valid Loss: 3.3604

Training complete in 5h 9m 37s
Best Loss  2.10845517892442