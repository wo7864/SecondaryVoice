import sys
sys.path.append('assem-vc/')
sys.path.append('assem-vc/hifi-gan/')

from wav2mel import wav2mel
from mel2wav import mel2wav

if __name__ == "__main__":
    src_audio_num = 0
    target_audio_path = 'example/p225_005-22k.wav'
    save_path = target_audio_path.split('/')[-1].split('.')[0] + '.wav'
    mel = wav2mel(target_audio_path)
    audio = mel2wav(src_audio_num, mel, save_path)
    print(f'saved {save_path}')