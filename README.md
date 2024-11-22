# 요일별 주차 단속과 소화전 앞 불법주차 디텍팅 및 음성안내

### Seongeun Lee, Juho Kim, Hyein Bae

간단한 설명: YOLOv5를 이용하여 번호판과 번호판 글자를 인식합니다. 번호판마다 차량종류(대형,소형,응급차 등등)를 인식합니다. YOLOv5로만 글자를 다 인식하게 하여 불필요한 추가학습은 제거 했습니다. 소화전 또한 같이 탐색 하였습니다. 인식된 번호판을 TEXT로 출력하였습니다.
[YOLOv5](https://github.com/ultralytics/yolov5)
음성안내는 카카오에서 만든 VITs TTS모델을 이용하여 학습하였습니다.[참고1.VITs](https://github.com/jaywalnut310/vits)과 [참고2.VITs](https://github.com/ouor/vits?tab=readme-ov-file)를 참고해주시기 바랍니다.

목소리는 Korean Single Speaker Speech Dataset(KSS)으로 학습하였습니다.
[KSS_Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)데이터셋은 이곳에서 다운로드 하였습니다.

```sh
conda create -n detect_car python==3.9
pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

훈련파일은 드라이브로 제공하겠습니다.
