args:{'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'T_0': 10, 'lr': 0.001, 'min_lr': 1e-06, 'weight_decay': 1e-05, 'batch_size': 10, 'n_fold': 10, 'smoothing': 0.3, 'net_architect': 'VisionTransformer', 'compart_name': 'resized', 'sequence': ['t1', 't2', 't1ce', 'flair'], 'data_key_list': ['sex', 'age', 'IDH', 'MGMT', 'GBL', 'EOR', 'duration_death', 'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion'], 'root_dir': '/mnt/hdd3/mskim/GBL', 'data_dir': '/mnt/hdd3/mskim/GBL/data', 'label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels', 'proc_label_dir': '/mnt/hdd3/mskim/GBL/data/label/surv_labels/proc_labels', 'exp_dir': '/mnt/hdd3/mskim/GBL/code/experiment', 'dataset_name': ['SNUH']}
Using VisionTransformer
n_intervals: 19
Train dataset_name:SNUH
Training on GPU 0
Setting seed:123456
dataset:SNUH for training
df_dataset.shape:(1134, 11)
df_label_dataset_list.shape:(1134, 11)
filtering before GBL; 1134 cases
filtering after GBL; 654 cases
events before 1yr:
1141
events after 1yr:
828
dataset:severance for training
ext_df_dataset.shape:(132, 11)
filtering before GBL; 132 cases
filtering after GBL; 104 cases
events before 1yr:
81
events after 1yr:
27
df.index.values:654
img_label_comm_list:652
dataset_name:SNUH, 652
SNUH df.shape: (652, 11)
n_intervals: 19
df.index.values:104
img_label_comm_list:104
dataset_name:severance, 104
severance df.shape: (104, 11)
n_intervals: 19
Show Vision Transformer Architect:OrderedDict([('patch_embed', PatchEmbed(
  (proj): Conv3d(4, 768, kernel_size=(16, 16, 16), stride=(16, 16, 16))
  (norm): Identity()
)), ('pos_drop', Dropout(p=0.0, inplace=False)), ('blocks', ModuleList(
  (0-11): 12 x Block(
    (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): Attention(
      (qkv): Linear(in_features=768, out_features=2304, bias=True)
      (q_norm): Identity()
      (k_norm): Identity()
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ls1): Identity()
    (drop_path1): Identity()
    (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (mlp): Mlp(
      (fc1): Linear(in_features=768, out_features=3072, bias=True)
      (act): GELU(approximate='none')
      (drop1): Dropout(p=0.0, inplace=False)
      (norm): Identity()
      (fc2): Linear(in_features=3072, out_features=768, bias=True)
      (drop2): Dropout(p=0.0, inplace=False)
    )
    (ls2): Identity()
    (drop_path2): Identity()
  )
)), ('cls_norm', LayerNorm((768,), eps=1e-06, elementwise_affine=True)), ('head', Linear(in_features=768, out_features=19, bias=True))])
Vision Transformer Layer List:['patch_embed', 'pos_drop', 'blocks', 'cls_norm', 'head']

3D MRI GBM Size:torch.Size([10, 4, 224, 224, 224])
Batch,Embed_dim,patch_num:torch.Size([10, 768, 14, 14, 14])
After Patch Embedding:torch.Size([10, 2744, 768])
Positional Embedding Vector Size():torch.Size([1, 2745, 768])
Transformer Encoder input:torch.Size([10, 2745, 768])

===Training 1yr, GBL===

[586 rows x 39 columns]
num of train_data: 586
num of valid_data: 66
Epoch 1/200
----------
phase:train
train Loss: 41.8349
phase:valid
valid Loss: 36.3588

Epoch 2/200
----------
phase:train
train Loss: 36.9999
phase:valid
valid Loss: 31.5723

Epoch 3/200
----------
phase:train
train Loss: 33.3550
phase:valid
valid Loss: 29.4984

Epoch 4/200
----------
phase:train
train Loss: 31.7729
phase:valid
valid Loss: 28.1134

Epoch 5/200
----------
phase:train
train Loss: 27.7863
phase:valid
valid Loss: 21.4939

Epoch 6/200
----------
phase:train
train Loss: 24.1890
phase:valid
valid Loss: 20.5157

Epoch 7/200
----------
phase:train
train Loss: 23.5275
phase:valid
valid Loss: 20.0264

Epoch 8/200
----------
phase:train
train Loss: 23.1116
phase:valid
valid Loss: 19.6839

Epoch 9/200
----------
phase:train
train Loss: 22.8915
phase:valid
valid Loss: 19.5695

Epoch 10/200
----------
phase:train
train Loss: 22.8169
phase:valid
valid Loss: 19.5256

Epoch 11/200
----------
phase:train
train Loss: 22.7925
phase:valid
valid Loss: 19.5242

Epoch 12/200
----------
phase:train
train Loss: 22.7749
phase:valid
valid Loss: 19.4834

Epoch 13/200
----------
phase:train
train Loss: 22.6927
phase:valid
valid Loss: 19.3444

Epoch 14/200
----------
phase:train
train Loss: 22.4868
phase:valid
valid Loss: 19.0332

Epoch 15/200
----------
phase:train
train Loss: 22.1154
phase:valid
valid Loss: 18.5922

Epoch 16/200
----------
phase:train
train Loss: 21.6010
phase:valid
valid Loss: 17.9378

Epoch 17/200
----------
phase:train
train Loss: 20.9549
phase:valid
valid Loss: 17.2487

Epoch 18/200
----------
phase:train
train Loss: 20.2424
phase:valid
valid Loss: 16.4462

Epoch 19/200
----------
phase:train
train Loss: 19.4928
phase:valid
valid Loss: 15.6415

Epoch 20/200
----------
phase:train
train Loss: 18.7454
phase:valid
valid Loss: 14.8600

Epoch 21/200
----------
phase:train
train Loss: 18.0346
phase:valid
valid Loss: 14.1512

Epoch 22/200
----------
phase:train
train Loss: 17.3788
phase:valid
valid Loss: 13.4906

Epoch 23/200
----------
phase:train
train Loss: 16.7971
phase:valid
valid Loss: 12.9073

Epoch 24/200
----------
phase:train
train Loss: 16.3075
phase:valid
valid Loss: 12.4432

Epoch 25/200
----------
phase:train
train Loss: 15.9030
phase:valid
valid Loss: 12.0647

Epoch 26/200
----------
phase:train
train Loss: 15.5930
phase:valid
valid Loss: 11.8001

Epoch 27/200
----------
phase:train
train Loss: 15.3748
phase:valid
valid Loss: 11.6024

Epoch 28/200
----------
phase:train
train Loss: 15.2307
phase:valid
valid Loss: 11.4948

Epoch 29/200
----------
phase:train
train Loss: 15.1542
phase:valid
valid Loss: 11.4428

Epoch 30/200
----------
phase:train
train Loss: 15.1228
phase:valid
valid Loss: 11.4312

Epoch 31/200
----------
phase:train
train Loss: 15.1163
phase:valid
valid Loss: 11.4308

Epoch 32/200
----------
phase:train
train Loss: 15.1097
phase:valid
valid Loss: 11.4167

Epoch 33/200
----------
phase:train
train Loss: 15.0783
phase:valid
valid Loss: 11.3618

Epoch 34/200
----------
phase:train
train Loss: 15.0023
phase:valid
valid Loss: 11.2550

Epoch 35/200
----------
phase:train
train Loss: 14.8622
phase:valid
valid Loss: 11.0583

Epoch 36/200
----------
phase:train
train Loss: 14.6512
phase:valid
valid Loss: 10.8105

Epoch 37/200
----------
phase:train
train Loss: 14.3705
phase:valid
valid Loss: 10.4776

Epoch 38/200
----------
phase:train
train Loss: 14.0313
phase:valid
valid Loss: 10.0819

Epoch 39/200
----------
phase:train
train Loss: 13.6509
phase:valid
valid Loss: 9.6616

Epoch 40/200
----------
phase:train
train Loss: 13.2517
phase:valid
valid Loss: 9.2443

Epoch 41/200
----------
phase:train
train Loss: 12.8540
phase:valid
valid Loss: 8.8245

Epoch 42/200
----------
phase:train
train Loss: 12.4764
phase:valid
valid Loss: 8.4487

Epoch 43/200
----------
phase:train
train Loss: 12.1363
phase:valid
valid Loss: 8.1136

Epoch 44/200
----------
phase:train
train Loss: 11.8370
phase:valid
valid Loss: 7.8087

Epoch 45/200
----------
phase:train
train Loss: 11.5921
phase:valid
valid Loss: 7.5909

Epoch 46/200
----------
phase:train
train Loss: 11.3985
phase:valid
valid Loss: 7.4109

Epoch 47/200
----------
phase:train
train Loss: 11.2610
phase:valid
valid Loss: 7.2936

Epoch 48/200
----------
phase:train
train Loss: 11.1698
phase:valid
valid Loss: 7.2216

Epoch 49/200
----------
phase:train
train Loss: 11.1202
phase:valid
valid Loss: 7.1887

Epoch 50/200
----------
phase:train
train Loss: 11.0997
phase:valid
valid Loss: 7.1794

Epoch 51/200
----------
phase:train
train Loss: 11.0952
phase:valid
valid Loss: 7.1790

Epoch 52/200
----------
phase:train
train Loss: 11.0913
phase:valid
valid Loss: 7.1704

Epoch 53/200
----------
phase:train
train Loss: 11.0703
phase:valid
valid Loss: 7.1362

Epoch 54/200
----------
phase:train
train Loss: 11.0189
phase:valid
valid Loss: 7.0589

Epoch 55/200
----------
phase:train
train Loss: 10.9250
phase:valid
valid Loss: 6.9339

Epoch 56/200
----------
phase:train
train Loss: 10.7821
phase:valid
valid Loss: 6.7552

Epoch 57/200
----------
phase:train
train Loss: 10.5904
phase:valid
valid Loss: 6.5288

Epoch 58/200
----------
phase:train
train Loss: 10.3553
phase:valid
valid Loss: 6.2711

Epoch 59/200
----------
phase:train
train Loss: 10.0900
phase:valid
valid Loss: 5.9619

Epoch 60/200
----------
phase:train
train Loss: 9.8048
phase:valid
valid Loss: 5.6533

Epoch 61/200
----------
phase:train
train Loss: 9.5159
phase:valid
valid Loss: 5.3454

Epoch 62/200
----------
phase:train
train Loss: 9.2360
phase:valid
valid Loss: 5.0596

Epoch 63/200
----------
phase:train
train Loss: 8.9806
phase:valid
valid Loss: 4.8020

Epoch 64/200
----------
phase:train
train Loss: 8.7557
phase:valid
valid Loss: 4.5861

Epoch 65/200
----------
phase:train
train Loss: 8.5666
phase:valid
valid Loss: 4.4077

Epoch 66/200
----------
phase:train
train Loss: 8.4189
phase:valid
valid Loss: 4.2742

Epoch 67/200
----------
phase:train
train Loss: 8.3108
phase:valid
valid Loss: 4.1801

Epoch 68/200
----------
phase:train
train Loss: 8.2401
phase:valid
valid Loss: 4.1273

Epoch 69/200
----------
phase:train
train Loss: 8.2016
phase:valid
valid Loss: 4.0994

Epoch 70/200
----------
phase:train
train Loss: 8.1851
phase:valid
valid Loss: 4.0924

Epoch 71/200
----------
phase:train
train Loss: 8.1818
phase:valid
valid Loss: 4.0921

Epoch 72/200
----------
phase:train
train Loss: 8.1784
phase:valid
valid Loss: 4.0849

Epoch 73/200
----------
phase:train
train Loss: 8.1623
phase:valid
valid Loss: 4.0586

Epoch 74/200
----------
phase:train
train Loss: 8.1226
phase:valid
valid Loss: 3.9992

Epoch 75/200
----------
phase:train
train Loss: 8.0487
phase:valid
valid Loss: 3.8986

Epoch 76/200
----------
phase:train
train Loss: 7.9336
phase:valid
valid Loss: 3.7622

Epoch 77/200
----------
phase:train
train Loss: 7.7809
phase:valid
valid Loss: 3.5715

Epoch 78/200
----------
phase:train
train Loss: 7.5916
phase:valid
valid Loss: 3.3571

Epoch 79/200
----------
phase:train
train Loss: 7.3736
phase:valid
valid Loss: 3.1127

Epoch 80/200
----------
phase:train
train Loss: 7.1392
phase:valid
valid Loss: 2.8545

Epoch 81/200
----------
phase:train
train Loss: 6.8988
phase:valid
valid Loss: 2.6007

Epoch 82/200
----------
phase:train
train Loss: 6.6668
phase:valid
valid Loss: 2.3571

Epoch 83/200
----------
phase:train
train Loss: 6.4482
phase:valid
valid Loss: 2.1486

Epoch 84/200
----------
phase:train
train Loss: 6.2555
phase:valid
valid Loss: 1.9560

Epoch 85/200
----------
phase:train
train Loss: 6.0938
phase:valid
valid Loss: 1.8039

Epoch 86/200
----------
phase:train
train Loss: 5.9682
phase:valid
valid Loss: 1.6824

Epoch 87/200
----------
phase:train
train Loss: 5.8719
phase:valid
valid Loss: 1.6051

Epoch 88/200
----------
phase:train
train Loss: 5.8121
phase:valid
valid Loss: 1.5570

Epoch 89/200
----------
phase:train
train Loss: 5.7785
phase:valid
valid Loss: 1.5310

Epoch 90/200
----------
phase:train
train Loss: 5.7633
phase:valid
valid Loss: 1.5261

Epoch 91/200
----------
phase:train
train Loss: 5.7612
phase:valid
valid Loss: 1.5258

Epoch 92/200
----------
phase:train
train Loss: 5.7580
phase:valid
valid Loss: 1.5196

Epoch 93/200
----------
phase:train
train Loss: 5.7443
phase:valid
valid Loss: 1.4954

Epoch 94/200
----------
phase:train
train Loss: 5.7085
phase:valid
valid Loss: 1.4464

Epoch 95/200
----------
phase:train
train Loss: 5.6448
phase:valid
valid Loss: 1.3564

Epoch 96/200
----------
phase:train
train Loss: 6.5456
phase:valid
valid Loss: -0.7024

Epoch 97/200
----------
phase:train
train Loss: 3.8102
phase:valid
valid Loss: -1.1609

Epoch 98/200
----------
phase:train
train Loss: 3.5337
phase:valid
valid Loss: -1.3998

Epoch 99/200
----------
phase:train
train Loss: 3.3162
phase:valid
valid Loss: -1.6199

Epoch 100/200
----------
phase:train
train Loss: 3.1109
phase:valid
valid Loss: -1.8236

Epoch 101/200
----------
phase:train
train Loss: 2.9170
phase:valid
valid Loss: -2.0182

Epoch 102/200
----------
phase:train
train Loss: 2.7376
phase:valid
valid Loss: -2.2020

Epoch 103/200
----------
phase:train
train Loss: 2.5748
phase:valid
valid Loss: -2.3570

Epoch 104/200
----------
phase:train
train Loss: 2.4359
phase:valid
valid Loss: -2.4928

Epoch 105/200
----------
phase:train
train Loss: 2.3194
phase:valid
valid Loss: -2.6004

Epoch 106/200
----------
phase:train
train Loss: 2.2290
phase:valid
valid Loss: -2.6822

Epoch 107/200
----------
phase:train
train Loss: 2.1631
phase:valid
valid Loss: -2.7381

Epoch 108/200
----------
phase:train
train Loss: 2.1198
phase:valid
valid Loss: -2.7722

Epoch 109/200
----------
phase:train
train Loss: 2.0956
phase:valid
valid Loss: -2.7880

Epoch 110/200
----------
phase:train
train Loss: 2.0859
phase:valid
valid Loss: -2.7923

Epoch 111/200
----------
phase:train
train Loss: 2.0837
phase:valid
valid Loss: -2.7924

Epoch 112/200
----------
phase:train
train Loss: 2.0816
phase:valid
valid Loss: -2.7970

Epoch 113/200
----------
phase:train
train Loss: 2.0716
phase:valid
valid Loss: -2.8140

Epoch 114/200
----------
phase:train
train Loss: 2.0463
phase:valid
valid Loss: -2.8511

Epoch 115/200
----------
phase:train
train Loss: 1.9995
phase:valid
valid Loss: -2.9121

Epoch 116/200
----------
phase:train
train Loss: 1.9280
phase:valid
valid Loss: -3.0012

Epoch 117/200
----------
phase:train
train Loss: 1.8306
phase:valid
valid Loss: -3.1155

Epoch 118/200
----------
phase:train
train Loss: 1.7105
phase:valid
valid Loss: -3.2525

Epoch 119/200
----------
phase:train
train Loss: 1.5727
phase:valid
valid Loss: -3.4046

Epoch 120/200
----------
phase:train
train Loss: 1.4246
phase:valid
valid Loss: -3.5657

Epoch 121/200
----------
phase:train
train Loss: 1.2718
phase:valid
valid Loss: -3.7265

Epoch 122/200
----------
phase:train
train Loss: 1.1233
phase:valid
valid Loss: -3.8785

Epoch 123/200
----------
phase:train
train Loss: 0.9844
phase:valid
valid Loss: -4.0178

Epoch 124/200
----------
phase:train
train Loss: 0.8618
phase:valid
valid Loss: -4.1373

Epoch 125/200
----------
phase:train
train Loss: 0.7575
phase:valid
valid Loss: -4.2352

Epoch 126/200
----------
phase:train
train Loss: 0.6749
phase:valid
valid Loss: -4.3100

Epoch 127/200
----------
phase:train
train Loss: 0.6147
phase:valid
valid Loss: -4.3621

Epoch 128/200
----------
phase:train
train Loss: 0.5750
phase:valid
valid Loss: -4.3927

Epoch 129/200
----------
phase:train
train Loss: 0.5531
phase:valid
valid Loss: -4.4073

Epoch 130/200
----------
phase:train
train Loss: 0.5441
phase:valid
valid Loss: -4.4113

Epoch 131/200
----------
phase:train
train Loss: 0.5421
phase:valid
valid Loss: -4.4115

Epoch 132/200
----------
phase:train
train Loss: 0.5401
phase:valid
valid Loss: -4.4157

Epoch 133/200
----------
phase:train
train Loss: 0.5312
phase:valid
valid Loss: -4.4309

Epoch 134/200
----------
phase:train
train Loss: 0.5081
phase:valid
valid Loss: -4.4649

Epoch 135/200
----------
phase:train
train Loss: 0.4650
phase:valid
valid Loss: -4.5216

Epoch 136/200
----------
phase:train
train Loss: 0.3989
phase:valid
valid Loss: -4.6045

Epoch 137/200
----------
phase:train
train Loss: 0.3089
phase:valid
valid Loss: -4.7121

Epoch 138/200
----------
phase:train
train Loss: 0.1971
phase:valid
valid Loss: -4.8390

Epoch 139/200
----------
phase:train
train Loss: 0.0680
phase:valid
valid Loss: -4.9840

Epoch 140/200
----------
phase:train
train Loss: -0.0737
phase:valid
valid Loss: -5.1383

Epoch 141/200
----------
phase:train
train Loss: -0.2205
phase:valid
valid Loss: -5.2925

Epoch 142/200
----------
phase:train
train Loss: -0.3647
phase:valid
valid Loss: -5.4410

Epoch 143/200
----------
phase:train
train Loss: -0.5007
phase:valid
valid Loss: -5.5792

Epoch 144/200
----------
phase:train
train Loss: -0.6221
phase:valid
valid Loss: -5.6978

Epoch 145/200
----------
phase:train
train Loss: -0.7250
phase:valid
valid Loss: -5.7954

Epoch 146/200
----------
phase:train
train Loss: -0.8075
phase:valid
valid Loss: -5.8697

Epoch 147/200
----------
phase:train
train Loss: -0.8674
phase:valid
valid Loss: -5.9218

Epoch 148/200
----------
phase:train
train Loss: -0.9072
phase:valid
valid Loss: -5.9531

Epoch 149/200
----------
phase:train
train Loss: -0.9295
phase:valid
valid Loss: -5.9677

Epoch 150/200
----------
phase:train
train Loss: -0.9384
phase:valid
valid Loss: -5.9716

Epoch 151/200
----------
phase:train
train Loss: -0.9405
phase:valid
valid Loss: -5.9718

Epoch 152/200
----------
phase:train
train Loss: -0.9423
phase:valid
valid Loss: -5.9758

Epoch 153/200
----------
phase:train
train Loss: -0.9514
phase:valid
valid Loss: -5.9914

Epoch 154/200
----------
phase:train
train Loss: -0.9742
phase:valid
valid Loss: -6.0244

Epoch 155/200
----------
phase:train
train Loss: -1.0165
phase:valid
valid Loss: -6.0806

Epoch 156/200
----------
phase:train
train Loss: -1.0807
phase:valid
valid Loss: -6.1612

Epoch 157/200
----------
phase:train
train Loss: -1.1700
phase:valid
valid Loss: -6.2667

Epoch 158/200
----------
phase:train
train Loss: -1.2807
phase:valid
valid Loss: -6.3954

Epoch 159/200
----------
phase:train
train Loss: -1.4097
phase:valid
valid Loss: -6.5397

Epoch 160/200
----------
phase:train
train Loss: -1.5508
phase:valid
valid Loss: -6.6943

Epoch 161/200
----------
phase:train
train Loss: -1.6990
phase:valid
valid Loss: -6.8489

Epoch 162/200
----------
phase:train
train Loss: -1.8438
phase:valid
valid Loss: -7.0006

Epoch 163/200
----------
phase:train
train Loss: -1.9818
phase:valid
valid Loss: -7.1384

Epoch 164/200
----------
phase:train
train Loss: -2.1042
phase:valid
valid Loss: -7.2609

Epoch 165/200
----------
phase:train
train Loss: -2.2096
phase:valid
valid Loss: -7.3601

Epoch 166/200
----------
phase:train
train Loss: -2.2928
phase:valid
valid Loss: -7.4360

Epoch 167/200
----------
phase:train
train Loss: -2.3541
phase:valid
valid Loss: -7.4884

Epoch 168/200
----------
phase:train
train Loss: -2.3944
phase:valid
valid Loss: -7.5202

Epoch 169/200
----------
phase:train
train Loss: -2.4167
phase:valid
valid Loss: -7.5344

Epoch 170/200
----------
phase:train
train Loss: -2.4260
phase:valid
valid Loss: -7.5386

Epoch 171/200
----------
phase:train
train Loss: -2.4281
phase:valid
valid Loss: -7.5387

Epoch 172/200
----------
phase:train
train Loss: -2.4300
phase:valid
valid Loss: -7.5428

Epoch 173/200
----------
phase:train
train Loss: -2.4391
phase:valid
valid Loss: -7.5580

Epoch 174/200
----------
phase:train
train Loss: -2.4619
phase:valid
valid Loss: -7.5916

Epoch 175/200
----------
phase:train
train Loss: -2.5041
phase:valid
valid Loss: -7.6469

Epoch 176/200
----------
phase:train
train Loss: -2.5688
phase:valid
valid Loss: -7.7289

Epoch 177/200
----------
phase:train
train Loss: -2.6576
phase:valid
valid Loss: -7.8359

Epoch 178/200
----------
phase:train
train Loss: -2.7684
phase:valid
valid Loss: -7.9610

Epoch 179/200
----------
phase:train
train Loss: -2.8973
phase:valid
valid Loss: -8.1072

Epoch 180/200
----------
phase:train
train Loss: -3.0393
phase:valid
valid Loss: -8.2638

Epoch 181/200
----------
phase:train
train Loss: -3.1873
phase:valid
valid Loss: -8.4197

Epoch 182/200
----------
phase:train
train Loss: -3.3326
phase:valid
valid Loss: -8.5713

Epoch 183/200
----------
phase:train
train Loss: -3.4718
phase:valid
valid Loss: -8.7098

Epoch 184/200
----------
phase:train
train Loss: -3.5942
phase:valid
valid Loss: -8.8344

Epoch 185/200
----------
phase:train
train Loss: -3.7003
phase:valid
valid Loss: -8.9311

Epoch 186/200
----------
phase:train
train Loss: -3.7844
phase:valid
valid Loss: -9.0086

Epoch 187/200
----------
phase:train
train Loss: -3.8461
phase:valid
valid Loss: -9.0625

Epoch 188/200
----------
phase:train
train Loss: -3.8865
phase:valid
valid Loss: -9.0924

Epoch 189/200
----------
phase:train
train Loss: -3.9095
phase:valid
valid Loss: -9.1080

Epoch 190/200
----------
phase:train
train Loss: -3.9189
phase:valid
valid Loss: -9.1130

Epoch 191/200
----------
phase:train
train Loss: -3.9210
phase:valid
valid Loss: -9.1132

Epoch 192/200
----------
phase:train
train Loss: -3.9229
phase:valid
valid Loss: -9.1170

Epoch 193/200
----------
phase:train
train Loss: -3.9319
phase:valid
valid Loss: -9.1328

Epoch 194/200
----------
phase:train
train Loss: -3.9548
phase:valid
valid Loss: -9.1635

Epoch 195/200
----------
phase:train
train Loss: -3.9967
phase:valid
valid Loss: -9.2203

Epoch 196/200
----------
phase:train
train Loss: -4.0602
phase:valid
valid Loss: -9.3022

Epoch 197/200
----------
phase:train
train Loss: -4.1491
phase:valid
valid Loss: -9.4086

Epoch 198/200
----------
phase:train
train Loss: -4.2591
phase:valid
valid Loss: -9.5378

Epoch 199/200
----------
phase:train
train Loss: -4.3873
phase:valid
valid Loss: -9.6830

Epoch 200/200
----------
phase:train
train Loss: -4.5300
phase:valid
valid Loss: -9.8364

Training complete in 16h 6m 50s
Best Loss  -9.836423873901367

===Survival Analysis===
Testing on GPU 1

df_DL_score_test.shape:(104, 20)
104
duration_test.shape:(104,)
oneyr_survs_test.shape:(104,)
event_test.shape:(104,)

Original C-index for valid: 0.4888
95% CI for C-index for valid: (0.3807, 0.6029)
