# 사용자 정의 데이터 세트로 YOLO-NAS 학습시키기

이 리포지토리는 사용자 정의 데이터 세트로 YOLO-NAS를 학습시키는 방법을 설명합니다. LearnOpenCV의 포스트를 참고하여 작성되었습니다.

## 환경설정

### 패키지 설치

1. `super-gradients` 패키지 설치
2. GPU 사용을 위해 CUDA와 호환되는 PyTorch 설치

## 데이터 세트

YOLO-NAS 포맷의 데이터 세트를 준비합니다. 이 예제에서는 Roboflow에서 제공하는 안전모 데이터 세트를 사용했습니다.

## 구현

### Windows 사용자를 위한 참고사항

Windows 환경에서는 `torch.multiprocessing.freeze_support()`를 사용하여 무한 재귀 문제를 방지합니다.

### 초매개변수 선언

- Epoch: 50
- Batch Size: 16

### 결과 이미지
![005358_jpg rf 213116ce2ac4cd3f071490db20eca141](https://github.com/zoid79/YOLO_NAS_Webcam/assets/87366543/0988fe8f-0485-4972-a3e4-d6012f3b1307)
