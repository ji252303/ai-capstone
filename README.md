# Dream in Guitar: AI 기타 강습

## 팀원
황민혁(조장), 백인수  

## 프로젝트 개요
이 프로젝트의 주목적은 누구든 집에서 쉽게 접할 수 있는 AI 기반 기타 강습을 제공하는 것입니다. 이를 통해 기타 연주의 꿈을 가진 이들에게 프로그램을 제공합니다.

## 주요 기능
- **튜토리얼 기능**: 기본 코드 운지법을 안내하고 사용자의 연주가 올바른지 실시간으로 확인하여 피드백을 제공합니다.
- **기본 연주 기능**: 사용자가 익힌 코드를 바탕으로 간단한 곡을 연주할 수 있으며, 연주 중 오류를 분석하여 피드백을 제공합니다.

## 시스템 구성도

![ai1](https://github.com/ji252303/ai-capstone/assets/95694431/cd45260e-4004-41ab-b0c7-f7650caf736d)

## 기술 스택
- **웹 서버(Flask)**: 회원가입, 로그인, 로그아웃 기능을 관리합니다.
- **기타 코드 인식**: 웹캠을 통해 사용자의 기타 코드를 실시간으로 인식하고 평가합니다.
- **피드백 시스템**: GPT-3.5 turbo를 사용하여 자연스러운 언어로 피드백을 제공합니다.

## 기술 소개

### 회원가입 및 로그인 기능
- **프레임워크**: Python의 Flask 웹 프레임워크 사용
- **회원가입**: 사용자 정보를 JSON 형태로 수집하여 데이터베이스에 저장
- **비밀번호 보안**: 비밀번호는 해시화하여 데이터베이스에 저장하여 보안 유지
- **로그인**: 사용자 정보를 세션에 저장하여 로그인 상태 유지
- **로그아웃**: 세션 정보를 삭제하고 메인 페이지로 리다이렉트

### 기타 코드 인식
- **기술**: 웹캠을 통한 실시간 기타 코드 인식
- **컴퓨터 비전**: mediapipe의 Hands 모듈을 사용한 손 위치 인식 및 코드 정확성 검증
- **컬러 피킹**: HSV 색공간을 활용하여 기타 지판의 코드 인식 정확도 향상

### 피드백 기능
- **AI 모델**: GPT-3.5 turbo를 활용한 피드백 생성
- **피드백 전달**: 손 위치가 부정확할 때 특정 프렛과 줄에 위치시키라는 지시 제공
- **정확한 연주 확인**: 모든 손가락이 올바른 위치에 있을 경우 '정확하게 잡았다'는 피드백 제공

### 클라이언트 - 튜토리얼 기능
- **웹캠**: 실시간으로 손의 위치를 확인하여 기타 코드의 정확성 판단
- **실시간 통신**: Socket.IO를 사용하여 서버와 실시간으로 통신하며 사용자에게 연속적인 피드백 제공
- **학습 진행**: 정확한 코드가 확인되면 다음 코드로 진행, 사용자에게 지속적인 가이드 제공

## 팀원 역할
- **황민혁(조장)**: 프로젝트 리더, 데이터 수집 및 전처리, CNN 모델을 이용한 음성 인식 개발.
- **백인수**: 컬러피킹을 통한 기타 지판 인식, 기타 코드 인식 알고리즘 개발.

## 개발 환경 설정

파이썬 3.7 환경의 가상환경 구축하기:

    cd SmartClimbing_iot_capstone
    conda create -n env python=3.7
    conda activate env

필요한 패키지들을 설치

    pip install -r requirements.txt

### 데모 실행하기

실행:

    python main.py

## 시연 화면

![ai2](https://github.com/ji252303/ai-capstone/assets/95694431/8ddbb694-720c-4c8f-8e99-cb2f3d227996)

![ai3](https://github.com/ji252303/ai-capstone/assets/95694431/e81cb6d4-d5ed-4ec4-ba97-ec2c6e242cd5)

![ai4](https://github.com/ji252303/ai-capstone/assets/95694431/8b5ba263-a1ab-4079-9efd-536fc7b0498e)

![ai5](https://github.com/ji252303/ai-capstone/assets/95694431/47db720c-76bc-4591-8a86-45dc51d674ee)



## 참고 문헌
- [기타 지판 라벨링 이미지 데이터셋](https://universe.roboflow.com/hubert-drapeau-qt6ae/guitar-necks-detector)
- [기타 코드 라벨링 이미지 데이터셋](https://universe.roboflow.com/school-sps5k/chorddetection2.2)
- [Mediapipe 손 랜드마크 검출](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)




