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

from g2p_en import G2p
from datasets.text import Language
from modules.mel import mel_spectrogram
from pysptk import sptk

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

g2p = G2p()
print("Complete.")

def script2text(script):
    
    phoneme = g2p(script)
    converted = ['{']
    for x in phoneme:
        if x==' ':
            converted.append('}')
            converted.append('{')
        elif x=='-':
            continue
        else:
            converted.append(x)

    converted.append('}')
    phoneme = " ".join(str(x) for x in converted)
    phoneme = phoneme.replace(' }', '}').replace('{ ','{')
    phoneme = phoneme.replace('0','').replace('1','').replace('2','').replace('{\'}','\'').replace('{...}','...')
    return phoneme.replace(' {!}','!').replace(' {?}','?').replace(' {.}','.').replace(' {,}',',')


def get_f0(audio, f0_mean=None, f0_var=None, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=80, f0_max=880, harm_thresh=0.25, mel_fmin = 70.0):

        f0 = sptk.rapt(audio*32768, sampling_rate, hop_length, min=f0_min, max=f0_max, otype=2)

        f0 = np.clip(f0, 0, f0_max)

        index_nonzero = np.nonzero(f0)
        f0[index_nonzero] += 10.0
        f0 -= 10.0

        if f0_mean == None:
            f0_mean =  np.mean(f0[index_nonzero])
        if f0_var == None:
            f0_var =  np.std(f0[index_nonzero])

        f0[index_nonzero] = (f0[index_nonzero] - f0_mean) / f0_var

        return f0


def get_mel_and_f0(audiopath, f0_mean=None, f0_var=None):
    wav, sr = librosa.load(audiopath, sr=None, mono=True)
    assert sr == hp.audio.sampling_rate, \
        'sample mismatch: expected %d, got %d at %s' % (hp.audio.sampling_rate, sr, audiopath)
    wav = torch.from_numpy(wav)
    wav = wav.unsqueeze(0)

    # wav = wav * (0.99 / (torch.max(torch.abs(wav)) + 1e-7))

    mel = mel_spectrogram(wav, hp.audio.filter_length, hp.audio.n_mel_channels,
                            hp.audio.sampling_rate,
                            hp.audio.hop_length, hp.audio.win_length,
                            hp.audio.mel_fmin, hp.audio.mel_fmax, center=False)
    mel = mel.squeeze(0)
    wav = wav.cpu().numpy()[0]
    f0 = get_f0(wav, f0_mean, f0_var, hp.audio.sampling_rate,
                        hp.audio.filter_length, hp.audio.hop_length, hp.audio.f0_min,
                        hp.audio.f0_max, hp.audio.harm_thresh, hp.audio.mel_fmin)
    f0 = torch.from_numpy(f0)[None]
    f0 = f0[:, :mel.size(1)]

    return mel, f0

def getSourceData(audiopath, script=None):

    text = script2text(script)
    print(text)
    lang = Language(hp.data.lang, hp.data.text_cleaners)
    text_norm = torch.IntTensor(lang.text_to_sequence(text, hp.data.text_cleaners))
    mel, f0 = get_mel_and_f0(audiopath)
    return text_norm, mel, 0, 0, f0


def mel2wav(src_num, target_mel, save_path, script):    
    ############################ Source Data 준비

    # sourceloader = TextMelDataset(hp, 'assem-vc/datasets/inference_source','metadata_g2p.txt',train=False, use_f0s = True)

    # source_idx = src_num # 0 ~ len(source_metadata)-1
    # audio_path, text,_ = sourceloader.meta[source_idx]
    # x = sourceloader.__getitem__(source_idx)

    x = getSourceData(src_num, script)
    print(x[0].size())
    print(x[1].size())
    print(x[4].size())
    batch = text_mel_collate([x])

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
    sf.write(save_path, audio, 22050)   
    return audio

if __name__ == "__main__":
    target_mel = 'example/a001.npy'
    mel2wav('', target_mel)




