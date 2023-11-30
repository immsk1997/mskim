'''MNIST Image-Classification Vision Transformer'''

# Ref : https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/MNIST%20Classification%20-%20ViT.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import save_image

from torchsummary import summary

import spacy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import math
from PIL import Image
import glob
from IPython.display import display
from transformer_package.models import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(1234567)
np.random.seed(1234567)

BATCH_SIZE = 64
LR = 5e-5  # 0.00005
NUM_EPOCHES = 25

'''정규화, 텐서화'''
mean, std = (0.5,), (0.5,)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

'''MNIST train / test'''
trainset = datasets.MNIST('../data/MNIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.MNIST('../data/MNIST/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)\

'''초기값 설정'''
image_size = 28
channel_size = 1
patch_size = 7
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2

model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)
model

'''데이터,라벨 Shape 확인 --> 파이토치 torch.utils.data.Dataloader (데이터를 메모리에 올리는 것)에서 이미지,라벨 가져오기 + GPU에 올리기'''
for img, label in trainloader:
    img = img.to(device)
    label = label.to(device)
    
    print("Input Image Dimensions: {}".format(img.size()))
    print("Label Dimensions: {}".format(label.size()))
    print("-"*100)
    
    out = model(img)
    
    print("Output Dimensions: {}".format(out.size()))
    break

'''Optimizer, Loss Function 설정'''
criterion = nn.NLLLoss() # negative Log Likelihood -> log likelihood를 최소화 하는 확률값 찾는 함수
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)


"""모델 학습 과정 구현"""

'''train accuracy, train loss를 딕셔너리 키로 관리, 각 Value는 리스트 자료형을 통해 Append()'''
loss_hist = {}
loss_hist["train accuracy"] = []
loss_hist["train loss"] = []

for epoch in range(1, NUM_EPOCHES+1):
    model.train()
    
    epoch_train_loss = 0
        
    y_true_train = []
    y_pred_train = []
    
    '''파이토치는 실제 학습 단계에서 enumerate를 활용하여 index를 같이 가져오는게 많이 보임 '''     
    for batch_idx, (img, labels) in enumerate(trainloader):
        img = img.to(device)
        labels = labels.to(device)
        
        '''enumerate로 불러와서 image,label을 GPU에 올리는 이유 '''
        '''->라벨은 loss 산출할 때 사용하기 위해 GPU에 올리는 것'''
        '''->이미지는 실제 모델에 입력으로 들어가서 계산하기 위해'''
        
        preds = model(img)
        
        loss = criterion(preds, labels)
        
        '''optimizer.zero_grad : 파이토치는 학습 Loop를 돌 때 이상적으로 학습이 되려면, 항상 역전파 하기전에 gradients를 zero로 만들어주고 시작해야함'''
        '''--------------------> gradient를 0으로 안해주면 학습이 이상하게 진행됨'''
        optimizer.zero_grad() 
        loss.backward() # 역전파
        optimizer.step() # Global Optimal 찾아가도록 분포 이동  
        
        y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
        y_true_train.extend(labels.detach().tolist())
            
        epoch_train_loss += loss.item()
    
    loss_hist["train loss"].append(epoch_train_loss)
    
    '''zip() : 반복 가능한 iter 요소를 Tuple로 반환해줌 -> 여기서는 prediction, true값을 묶어서 Tuple로 반환'''
    total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
    total = len(y_pred_train)
    
    '''Metrics : Accuracy = y_true,y_pred가 일치한 Case에 대해 전체 샘플로 나눈 비율'''
    accuracy = total_correct * 100 / total 
    
    loss_hist["train accuracy"].append(accuracy)
    
    print("Epoch: {} Train mean loss: {:.3f}".format(epoch, epoch_train_loss))
    print("       Train Accuracy: ", accuracy, "%", "==", total_correct, "/", total)
    print("-" * 10)
    
'''Accuracy / loss 시각화'''

plt.plot(loss_hist["train accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(loss_hist["train loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

'''Model Inference (모델 추론) -> Test Data'''

with torch.no_grad(): # with torch.no_grad() : gradient 계산 비활성화 -> Test data를 통해 모델의 성능을 확인하는 단계
    
    model.eval() # model.eval() : 모델의 모든 레이어가 Evaluation Mode로 전환 -> Normalization, Dropout 비활성화
    
    y_true_test = []
    y_pred_test = []
    
    for batch_idx, (img, labels) in enumerate(testloader):
        img = img.to(device)
        label = label.to(device)
    
        preds = model(img)
        
        y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
        y_true_test.extend(labels.detach().tolist())
        
    total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x==y])
    total = len(y_pred_test)
    accuracy = total_correct * 100 / total
    
    print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)

