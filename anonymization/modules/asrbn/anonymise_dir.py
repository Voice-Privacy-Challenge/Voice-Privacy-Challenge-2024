#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Pierre Champion
Re-implementation of: https://arxiv.org/abs/2308.04455
some features are missing from the thesis (Speaker F0 norm + F0 quant and other), results will differ.
"""

import multiprocessing
from pathlib import Path
import wave
from tqdm import tqdm

import torch
import torchaudio
import kaldiio

from utils import read_kaldi_format, copy_data_dir, create_clean_dir, setup_logger
from .utils import float2pcm

logger = setup_logger(__name__)

class Wav(): # for f0 extraction
    def __init__(self, w):
        self.wav = w

class Dataset(torch.utils.data.Dataset):
    def __init__(self, id_wavs, get_f0_func):
        self.all_wavs = list(id_wavs.values())
        self.all_keys = list(id_wavs.keys())
        self.get_f0_func = get_f0_func

    def __len__(self):
        return len(self.all_wavs)

    def __getitem__(self, index):
        audio, freq = torchaudio.load(self.all_wavs[index])
        f0 = self.get_f0_func(Wav(audio))
        return {"utid": self.all_keys[index],
                "audio": audio,
                "f0": f0,
                "freq": freq}

def collate_fn(item_list):
    batch_size = len(item_list)

    data_list_audio = [i['audio'] for i in item_list]
    lengths_tensor_audio = torch.tensor([i.shape[-1] for i in data_list_audio])
    max_len_audio = torch.max(lengths_tensor_audio).item()
    output_audio = torch.zeros([batch_size, max_len_audio])
    for i in range(batch_size):
        cur = data_list_audio[i]
        cur_len = data_list_audio[i].shape[-1]
        output_audio[i, :cur_len] = cur.squeeze()

    data_list_f0 = [i['f0'] for i in item_list]
    lengths_tensor_f0 = torch.tensor([i.shape[-1] for i in data_list_f0])
    max_len_f0 = torch.max(lengths_tensor_f0).item()
    output_f0 = torch.zeros([batch_size, max_len_f0])
    for i in range(batch_size):
        cur = data_list_f0[i]
        cur_len = data_list_f0[i].shape[-1]
        output_f0[i, :cur_len] = cur.squeeze()

    utids = [i['utid'] for i in item_list]
    freqs = [i['freq'] for i in item_list]
    return output_audio, output_f0, lengths_tensor_audio, utids, freqs

def process_data(dataset_path: Path, anon_level: str, results_dir: Path, settings: dict, force_compute: bool):
    output_path = Path(str(dataset_path) + settings['anon_suffix'])
    device = settings.get("device", "cpu")
    batch_size = settings.get("batch_size", 4)
    single_spkid = settings.get("single_spkid", "6081")
    tag_version = settings.get("model_tag_version", "6081") 

    copy_data_dir(dataset_path, output_path)
    results_dir = output_path / results_dir
    create_clean_dir(results_dir, force=force_compute)

    wav_scp = dataset_path / 'wav.scp'
    path_wav_scp_out = output_path / 'wav.scp'

    model = torch.hub.load("deep-privacy/SA-toolkit", "anonymization",
                           tag_version=tag_version,
                           trust_repo=True)
    model.to(device)
    model.eval()

    @torch.no_grad()
    def process_wav(utid, freq, audio, f0, original_len):

        freq = freq[0] # assume all freq = in same batch (and so dataset)
        audio = audio.to(device)

        # anonymize
        model.set_f0(f0.to(device)) # CPU extracted in by Dataloader (num_workers)
        wav_conv = model.convert(audio, target=single_spkid)
        wav_conv = wav_conv.cpu()

        def parallel_write():
            for i in range(wav_conv.shape[0]):
                wav = wav_conv[i]
                if len(wav.shape) > 1:
                    wav = wav[:, :original_len[i]]
                signal = float2pcm(wav.numpy())
                # write to buffer
                u = utid[i]
                output_file = results_dir / f'{u}.wav'
                with output_file.open('wb') as file:
                    with wave.open(file, 'wb') as stream:
                        stream.setframerate(freq)
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.writeframes(signal)
        p = multiprocessing.Process(target=parallel_write, args=())
        p.start()

        return p

    wavs = read_kaldi_format(wav_scp)
    nj = multiprocessing.cpu_count()
    nj = min(nj, 18)
    p = None

    with open(path_wav_scp_out, 'wt', encoding='utf-8') as writer:
        filtered_wavs = {}
        for u, file in wavs.items():
            output_file = results_dir / f'{u}.wav'
            if output_file.exists() and not force_compute:
                logger.debug(f'File {output_file} already exists')
                writer.writelines(f"{u} {output_file}\n")
            else:
                filtered_wavs[u] = file

        data_loader = torch.utils.data.DataLoader(Dataset(filtered_wavs, model.get_f0), batch_size=batch_size, num_workers=nj, collate_fn=collate_fn)
        for audio, f0, original_len, utid, freq in tqdm(data_loader):
            p = process_wav(utid, freq, audio, f0, original_len)
            for u in utid:
                output_file = results_dir / f'{u}.wav'
                writer.writelines(f"{u} {output_file}\n")
            torch.cuda.empty_cache()
    # wait for last p to write the anonymized audios
    if p:
        p.join()
    logger.info('Done')
