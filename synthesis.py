import sys
sys.path.append('assem-vc/')
sys.path.append('assem-vc/hifi-gan/')

from wav2mel import wav2mel
from mel2wav import mel2wav

if __name__ == "__main__":
    wav_path = 'example/p225_005-22k.wav'
    mel = wav2mel(wav_path)
    result = mel2wav('', mel)
    print(f'saved {result}')