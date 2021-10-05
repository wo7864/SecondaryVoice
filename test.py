import os
from glob import glob
import torch
from wav2mel import wav2mel

path = 'assem-vc/hifi-gan/LJSpeech-1.1/wavs'
save_path = 'assem-vc/hifi-gan/ft_dataset'
for p in glob(os.path.join(path, '*')):
    for p1 in glob(os.path.join(p, '*')):
        p2 = p1.rsplit('\\')
        new_file = os.path.join(p, 'test.wav')
        os.system(f'ffmpeg -y -i {p1} -ac 1 -ar 22050 {new_file}')
        os.system(f'mv {new_file} {p1}')
        os.system(f'rm {new_file}')
        mel = wav2mel(p1)
        torch.save(mel, os.path.join(save_path, p2[1], p2[2]+'.pt'))