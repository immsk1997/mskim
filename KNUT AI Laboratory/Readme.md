주제 : "초정밀 전기차 배터리 불량 유형을 분류하고 예측하는 이미지 분류 모델"


- 연구배경

미래 제조업 및 자동차 시장의 핵심은 "전동화", "무선화", "자율화" 라고 생각했습니다.

때마침 전기차에 대한 수요가 늘어감에 따라 2차전지에 대한 관심도 증대되었으며, 

이에 따라 앞으로 배터리의 품질이 더욱 중요해질 것이라고 생각했습니다.

이러한 생각을 기반으로 2D Image-Classification Task를 수행했습니다.


- 데이터셋 구성

"데이터는 서울에 위치한 모 반도체 업체에서 5가지 유형의 전기차 배터리 원본 이미지 500장을 제공받았습니다"

DataSet 1(전처리 x, 증강 x)
DataSet 2(전처리 o, 증강 x)
DataSet 3(전처리 x, 증강 o)
DataSet 4(전처리 o, 증강 o)


- 간략소개 

교내 AI 연구실 프로젝트 일환으로 Tensorflow 프레임워크의 Keras API, Keras Lib 활용

Standard CNN 알고리즘을 활용하여 Spatial dimension, Feature dimension을 혼합하여 학습하였습니다.

아시다시피 Standard CNN 같은 경우 Feature Map을 생성할 때 모든 채널의 정보가 담겨있습니다.

Pre-trained 사전학습 모델 RESNET 50, 101 , VGG 계열 모델 활용하여 정확도를 비교하였습니다.

Resnet 101 기반 우리 모델은 최종 Training Accuracy 최대 98.4% , Validation Accuracy 최대 91.12% 달성하였습니다.
