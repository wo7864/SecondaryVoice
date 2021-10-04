# AssemVC를 이용한 Voice Transfer 

[assem-vc Github Repository](https://github.com/mindslab-ai/assem-vc)

본 저장소는 assem-vc를 쉽게 사용 할 수 있는 코드를 제공합니다. 

## Clone Repository

```
git clone --recursive https://github.com/wo7864/SecondaryVoice
```



## Requirements

프로젝트 진행에 사용된 환경 설정은 아래와 같습니다.

- python 3.9.7
- torch 1.9.0
- pytorch-lightning 1.0.3

나머지 환경은 requirements.txt 파일을 참고해주세요.

```
pip install -r requirements.txt
```



## Pretrained Models

[다운로드](https://drive.google.com/drive/folders/1aIl8ObHxsmsFLXBz-y05jMBN4LrpQejm)

위 링크에서 아래 파일들을 다운로드합니다.

- hifi-gan_vctk_g_02600000
- assem-vc_pretrained.ckpt

다운로드한 파일은 `pretrained_models` 라는 폴더를 만들어 안에 저장합니다.



## How to Synthesis

가장 쉬운 방법은 `synthesis.py` 파일을 실행시키는 것 입니다.

```python
python synthesis.py
```

이 명령은 `example/p225_005-22k.wav` 음성을 `src:0번 발화`에 합성하여 `example/synthesis.wav` 로 저장합니다.

`src_audio_num`을 변경하면, 발화가 변경됩니다. 이는 0~2 까지 지정 할 수 있습니다.
곧 임의의 데이터를 src로 변경할 방법을 추가 할 계획입니다.

`target_audio_path`에 원하는 목소리의 음성 파일 경로를 입력 할 수 있습니다.

