# "초정밀 전기차 배터리 불량 유형을 분류하고 예측하는 이미지 분류 모델"

### **≪Achivement≫**
Title : 초정밀 배터리 단자 불량 유형 판정 딥러닝 모델 최적화


Prize : 한국교통대학교 산업공학과 컨퍼런스 대상 (공과대학장)


keywords : 2D-CNN, Image-classification

<img src ="https://github.com/immsk1997/mskim/blob/main/KNUT%20AI%20Laboratory/Paper/%ED%95%9C%EA%B5%AD%EA%B5%90%ED%86%B5%EB%8C%80%ED%95%99%EA%B5%90%20%EC%BB%A8%ED%8D%BC%EB%9F%B0%EC%8A%A4%20%EB%8C%80%EC%83%81.png" width="500" height="500">

## 연구배경


미래 제조업 및 자동차 시장의 핵심은 "전동화", "무선화", "자율화" 라고 생각했습니다.

때마침 전기차에 대한 수요가 늘어감에 따라 2차전지에 대한 관심도 증대되었으며, 

이에 따라 앞으로 배터리의 품질이 더욱 중요해질 것이라고 생각했습니다.

이러한 생각을 기반으로 2D Image-Classification Task를 수행했습니다.


## 데이터셋 구성

"데이터는 서울에 위치한 모 반도체 업체에서 5가지 유형의 전기차 배터리 원본 이미지 500장을 제공받았습니다"


DataSet 1(preprocess x, img_aug x)


DataSet 2(cv.crop, img_aug x)


DataSet 3(preprocess x, img_aug o)


DataSet 4(cv.crop, img_aug o)


## 간략소개 

교내 AI 연구실 프로젝트 일환으로써,
Tensorflow 프레임워크의 Keras API, Keras Lib 활용

Standard CNN 알고리즘을 활용하여 Spatial dimension, Feature dimension을 함께 학습하는 방식입니다.

Standard CNN 같은 경우 Feature Map을 생성할 때 공간정보와 모든 채널의 정보가 담겨있습니다.

Pre-trained 사전학습 모델 RESNET 50, 101, VGG 계열 모델 활용하여 정확도를 비교하였습니다.



## Results


Training Accuracy Max : 98.77% 


Training Accuracy Mean : 97.2%


Validation Accuracy Max : 91.13% 


Validation Accuracy Mean : 83.7%