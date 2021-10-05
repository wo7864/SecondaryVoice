import sys
sys.path.append('assem-vc/')
sys.path.append('assem-vc/hifi-gan/')

from stt import stt
from wav2mel import wav2mel
from mel2wav import mel2wav

if __name__ == "__main__":

    src_audio_num = 'example/p286_010-22k.wav'
    target_audio_path = 'example/a001.wav'
    save_path = target_audio_path.split('/')[-1].split('.')[0] + '.wav'

    script = stt(src_audio_num)
    mel = wav2mel(target_audio_path)
    audio = mel2wav(src_audio_num, mel, save_path, script)
    print(f'saved {save_path}')