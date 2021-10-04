import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('assem-vc/')
sys.path.append('assem-vc/hifi-gan/')

import os
import numpy as np
import json
import librosa
import torch
import random
import soundfile as sf

from omegaconf import OmegaConf
from env import AttrDict

from synthesizer import Synthesizer
from datasets import TextMelDataset, text_mel_collate
from models import Generator
from meldataset import MAX_WAV_VALUE

synthesizer_path = 'pretrained_models/assem-vc_pretrained.ckpt' # path of synthesizer(VC decoder) checkpoint
generator_path = 'pretrained_models/hifi-gan_vctk_g_02600000' # path of hifi-gan checkpoint
config_path = 'assem-vc/hifi-gan/config_v1.json' # path of hifi-gan's configuration
print("Loading Synthesizer's checkpoint...")
net=Synthesizer.load_from_checkpoint(synthesizer_path).cuda().eval()
net.freeze()

hp =net.hp
print("Complete.")
with open(config_path) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
torch.manual_seed(h.seed)
generator = Generator(h).cuda().eval()

print("Loading Generator's checkpoint...")

state_dict_g = torch.load(generator_path, map_location='cpu')
generator.load_state_dict(state_dict_g['generator'])
generator.remove_weight_norm()

print("Complete.")


def mel2wav(src_audio_path, target_mel):    
    ############################ Source Data 준비

    sourceloader = TextMelDataset(hp, 'assem-vc/datasets/inference_source','metadata_g2p.txt',train=False, use_f0s = True)

    source_idx = 1 # 0 ~ len(source_metadata)-1
    audio_path, text,_ = sourceloader.meta[source_idx]
    x = sourceloader.__getitem__(source_idx)
    batch = text_mel_collate([x])
    print("length of the source metadata is : ",len(sourceloader))

    ### target Data 준비
    if type(target_mel) == str:
        target_mel = np.load(target_mel)
    
    if type(target_mel) == np.ndarray:
        target_mel = torch.from_numpy(target_mel)
    target_mel= target_mel.unsqueeze(0)

    with torch.no_grad():
        text, mel_source, speakers, f0_padded, input_lengths, output_lengths, max_input_len, _ = batch
        text=text.cuda()
        mel_source = mel_source.cuda()
        mel_reference = target_mel.cuda()
        f0_padded = f0_padded.cuda()
        mel_predicted, alignment, residual = net.inference(text, mel_source, mel_reference, f0_padded)

    with torch.no_grad():
        y_g_hat = generator(mel_predicted.detach())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.detach().cpu().numpy().astype('int16')

    # 저장
    save_path = f'example/synthesis.wav'
    sf.write(save_path, audio, 22050)   
    return save_path

if __name__ == "__main__":
    target_mel = 'example/a001.npy'
    mel2wav('', target_mel)




