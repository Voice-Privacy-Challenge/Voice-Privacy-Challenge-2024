#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
@author: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
Audio Security and Privacy Group, EURECOM
modified version (N.T.)
"""
import functools
import hashlib
import itertools
import librosa
import librosa.core.spectrum
import logging
import multiprocessing
import numba
import numpy as np
import os
import random
import scipy
import scipy.signal
import shutil
import soundfile
import time
import wave

from copy import deepcopy
from itertools import repeat
from kaldiio import ReadHelper
from pathlib import Path
from tqdm import tqdm
from utils.data_io import read_kaldi_format

multiprocessing.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

def load_utt2spk(path):
    assert os.path.isfile(path), f'File does not exist {path}'
    table = np.genfromtxt(path, dtype='U')
    utt2spk = {utt: spk for utt, spk in table}
    return utt2spk

def hash_textstring(text):
    """number = hash_textstring(text)
    Hash text into a number. 
    Default hash() depends on PYTHONHASHSEED
    """
    # sha256 is deterministic, using the first 8Bytes (32bits)
    return np.abs(int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16))

def process_data(dataset_path: Path, anon_level: str, results_dir: Path, settings: dict):
    """
        Process data (must be in Kaldi format) in dataset_path with 
        B2 (McAdams coefficients-based) anonymization.

        Args:
            dataset_path (Path): Path to the dataset
            anon_level (str): Level of anonymization, either 'spk' or 'utt'
            results_dir (Path): Path to directory where results should be stored
            settings (dict): Settings for anonymization
    """
    utt2spk = None
    if anon_level == 'spk':
        utt2spk = load_utt2spk( dataset_path / 'utt2spk')
    basename = os.path.basename(dataset_path)
    output_path = Path(str(dataset_path) + settings['anon_suffix'])
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(dataset_path, output_path)
    results_dir = results_dir / basename
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    wav_scp = dataset_path / 'wav.scp'
    path_wav_scp_out = output_path / 'wav.scp'

    # get the number of utterances in the dataset
    wavs = read_kaldi_format(wav_scp)
    N = len(wavs)
 
    if anon_level == 'spk':
        # sample per speaker
        seeds = map(lambda x: hash_textstring(utt2spk[x]), wavs.keys())
        rngs = map(np.random.default_rng, seeds)
        mcadams_coeffs = map(lambda x: x.uniform(settings['mc_coeff_min'], settings['mc_coeff_max']), rngs)
    else:
        # sample per utterance
        rng = np.random.default_rng(hash_textstring('VPC2024'))
        mcadams_coeffs = rng.uniform(settings['mc_coeff_min'], settings['mc_coeff_max'], N)

    nj = multiprocessing.cpu_count()
    if 'nj' in settings:
        nj = settings['nj']

    with multiprocessing.Pool(processes=nj) as pool:
        with ReadHelper(f'scp:{wav_scp}') as reader:
            fn = functools.partial(process_wav, settings=settings, output_path=str(results_dir))
            scp_entries = pool.starmap(fn, tqdm(zip(mcadams_coeffs, reader), total=N), chunksize=10)
    with open(path_wav_scp_out, 'wt', encoding='utf-8') as writer:
        writer.writelines(scp_entries)    
    logger.info('Done')

def process_wav(mcadams, data, settings, output_path):
    output_path = Path(output_path)
    (utid, (freq, samples)) = data
    output_file = output_path / f'{utid}.wav'
    if output_file.exists():
        logger.debug(f'File {output_file} already exists')
        return f'{utid} {output_file}\n'

    # convert from int16 to float
    if samples.dtype == np.int16:
        samples = samples / (np.iinfo(np.int16).max + 1)
    
    samples = anonym_v2(freq=freq, samples=samples, 
        winLengthinms=settings['winLengthinms'],
        shiftLengthinms=settings['shiftLengthinms'], 
        lp_order=settings['n_coeffs'], mcadams=mcadams)

    # convert float to int16
    samples = (samples / np.max(np.abs(samples)) \
                * (np.iinfo(np.int16).max - 1)).astype(np.int16)

    # write to buffer
    with output_file.open('wb') as file:
        with wave.open(file, 'wb') as stream:
            stream.setframerate(freq)
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.writeframes(samples)
    
    #return the .scp entry
    return f'{utid} {output_file}\n'
            

def anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    eps = np.finfo(np.float32).eps
    samples = samples + eps
    
    # simulation parameters
    winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
    length_sig = len(samples)
    
    # fft processing parameters
    NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
    # anaysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int) # nr of complete frames   
    
    # carry out the overlap - add FFT processing
    sig_rec = np.zeros([length_sig]) # allocate output+'ringing' vector
    
    for m in np.arange(1, Nframes):
        # indices of the mth frame
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))    
        # windowed mth frame (other than rectangular window)
        frame = samples[index] * win 
        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)
        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
        #index of imaginary poles
        ind_imag = np.where(np.isreal(poles) == False)[0]
        #index of first imaginary poles
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]
        
        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        #new_angles = np.angle(poles[ind_imag_con])**path[m]
        
        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi        
        new_angles[np.where(new_angles <= 0)] = 0  
        
        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])            
        
        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))
        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc,np.array(1),frame)
        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]),a_lpc_new,res)
        frame_rec = frame_rec * win    

        outindex = np.arange(m * shift, m * shift + len(frame_rec))
        # overlap add
        sig_rec[outindex] = sig_rec[outindex] + frame_rec
        
    #sig_rec = (sig_rec / np.max(np.abs(sig_rec)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
    
    return sig_rec
    #scipy.io.wavfile.write(output_file, freq, np.float32(sig_rec))
    #awk -F'[/.]' '{print $5 " sox " $0 " -t wav -R -b 16 - |"}' > data/$dset$anon_data_suffix/wav.scp



def anonym_v2(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    """anonymized_data = anonym_v2(freq, samples, 
                                    winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8)
                                    
    input
    -----
      freq:              sampling frequency (Hz), scalar
      samples:           input waveform data,     np.array float32, (N, )
      winLengthinms:     analysis window length (ms), default 20ms
      shiftLengthinms:   analysis window shift (ms), default 10ms
      lp_order:          order of linear prediction analysis, default 20
      mcadams:           value of McAdam coef, default 0.8
    
    output
    ------
      anonymized_data:   anonymized data, np.array float32, (N, )
    """
    
    # to prevent numerical issue
    eps = np.finfo(np.float32).eps
    samples = samples + eps
    
    # short-time analysis parameters
    winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
    length_sig = len(samples)
    
    # fft processing parameters
    NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
    
    # anaysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int) # nr of complete frames   
    
    # framing
    frames = librosa.util.frame(samples, frame_length=winlen, hop_length=shift).T
    
    # windowing
    windowed_frames = frames * win
    
    # number of frames
    nframe = windowed_frames.shape[0]

    # LP analysis on all frames
    lpc_coefs = librosa.core.lpc(windowed_frames + eps, order=lp_order, axis=1)
    
    # get Poles for LP AR transfer function
    # tf2zpk only accepts a single transfunction function, 
    # we have to create a list 
    ar_poles = np.array([scipy.signal.tf2zpk(np.array([1]), x)[1] for x in lpc_coefs])

    def _mcadam_angle(poles, mcadams):
        """new_angles = _mcadam_angle(poles, mcadams)
        Adjust the angle of imaginary poles 
        
        poles: np.array, pole of AR function, (n, )
        mcadams: scalar, value of mcadams
        new_angles: np.array, (n, )
        """
        old_angles = np.angle(poles)
        # buffer to save
        new_angles = np.zeros_like(old_angles) + old_angles

        # real poles
        real_idx = ~np.isreal(poles)

        # imaginary with positive angle
        neg_idx = np.bitwise_and(real_idx, old_angles < 0.0)
        # imaginary with negative angle
        pos_idx = np.bitwise_and(real_idx, old_angles > 0.0)
        # 
        new_angles[neg_idx] = -((-np.angle(poles[neg_idx])) ** mcadams)
        # conjugate pair
        new_angles[pos_idx] = np.angle(poles[pos_idx]) ** mcadams

        # no need to constrain the range of the angle 
        # new_angles[np.where(new_angles >= np.pi)] = np.pi        
        # new_angles[np.where(new_angles <= 0)] = 0  

        return new_angles

    def _new_poles(old_poles, new_angles):
        """new_poles = _new_poles(old_poles, new_angles)
        create new poles using amplitude of original pole and new angle
        
        old_poles: np.array, pole of AR function, (n, )
        new_angles: np.array, (n, )
        """
        new_poles = np.zeros_like(old_poles)
        #new_poles[idx] = np.abs(old_poles[idx]) * np.exp(1j * new_angles[idx])
        #new_poles[idx+1] = np.abs(old_poles[idx] + 1) * np.exp(-1j * new_angles[idx])
        new_poles = np.abs(old_poles) * np.exp(1j * new_angles)
        return new_poles

    def _lpc_ana_syn(old_lpc_coef, new_lpc_coef, data):
        """new_data = _lpc_ana_syn(old_lpc_coef, new_lpc_coef, data)
        get excitation using old LPC, synthesize using new LPC coef
        
        old_lpc_coef: np.array, old LPC coef, (p, )
        new_lpc_coef: np.array, new LPC coef, (p, )
        data: np.array, data to be analyzed / synthesized (N, )
        """
        res = scipy.signal.lfilter(old_lpc_coef, np.array(1), data) 
        return scipy.signal.lfilter(np.array([1]), new_lpc_coef, res)

    pole_new_angles = _mcadam_angle(ar_poles, mcadams)
    poles_new = _new_poles(ar_poles, pole_new_angles)

    # reconstruct frame using new LPC coef
    recon_frames = [_lpc_ana_syn(lpc_coefs[x], np.real(np.poly(poles_new[x])), windowed_frames[x]) for x in np.arange(nframe)]
    recon_frames = np.stack(recon_frames, axis=0) * win

    # overlap-add
    anonymized_data = np.zeros_like(samples)
    librosa.core.spectrum.__overlap_add(anonymized_data, recon_frames.T, shift)
    
    # convert to int16
    #anonymized_data = (anonymized_data / np.max(np.abs(anonymized_data)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
    return anonymized_data

if __name__ == "__main__":
    print(__doc__)
