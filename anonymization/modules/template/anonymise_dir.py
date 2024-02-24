#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
Template example to show how to integrate an anonymization system to VPC2024.
"""

import wave
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from utils import read_kaldi_format, copy_data_dir, create_clean_dir, setup_logger, load_wav_from_scp

logger = setup_logger(__name__)


def process_data(dataset_path: Path, anon_level: str, results_dir: Path, settings: dict, force_compute: bool):
    # Path to the generated anonymized data-dir (data/dataset_template)
    output_path = Path(str(dataset_path) + settings['anon_suffix'])

    # Get some information from the configs/
    random_shape = settings.get("wav_shape", 42)

    ## From the source data-dir (./data/IEMOCAP_dev)
    ## Create the anonymized data-dir (./data/IEMOCAP_dev_template/)

    # Copy the utt2spk/spk2utt/text files from the source
    copy_data_dir(dataset_path, output_path)
    # Create the directory to store the anonymized data to. (./data/IEMOCAP_dev_template/anon_wav)
    # results_dir is configurable in the yaml, if force=True, anon_wav is emptied.
    results_dir = output_path / results_dir
    create_clean_dir(results_dir, force=force_compute)

    # Source scp file containing the list of audio files to anonymize. (./data/IEMOCAP_dev/wav.scp)
    wav_scp = dataset_path / 'wav.scp'
    # Target scp file containing the list of anonymized audio files. (./data/IEMOCAP_dev_template/wav.scp)
    path_wav_scp_out = output_path / 'wav.scp'

    # Your anonymization function
    def process_wav(uttid, freq, audio):

        ## Example of an input
        # logger.info(f"""{audio},   # Torch.tensor\n
        #             {audio.shape}, # Shape source audio\n
        #             {freq},        # Source audio frequency\n
        #             {uttid},       # wav.scp key (same as used in utt2spk/text/...)""")

        ## FAKE anonymization function ##
        anon_signal = torch.rand((1, random_shape))

        # Save results to ./data/IEMOCAP_dev_template/anon_wav/uttid.wav
        output_file = results_dir / f'{uttid}.wav'
        torchaudio.save(output_file, anon_signal, freq) # Prefer using the same freq as source


    # List of audio file to anonymize Dict of keys:uttid and values:audio file or kaldi like unix command
    wavs = read_kaldi_format(wav_scp)

    # Writer to write the uttid and paths to the anonymized audios
    with open(path_wav_scp_out, 'wt', encoding='utf-8') as writer:

        # Don't re-anonymize what has already being anonymized (force_compute=False else results_dir is new)
        filtered_wavs = {}
        for uttid, file in wavs.items():
            output_file = results_dir / f'{uttid}.wav'
            if output_file.exists() and not force_compute:
                logger.debug(f'File {output_file} already exists')
                writer.writelines(f"{uttid} {output_file}\n") # still rewrite to anon wav.scp (./data/IEMOCAP_dev_template/wav.scp)
            else:
                filtered_wavs[uttid] = file

        # Loop over the file to anonymize
        for uttid, file in tqdm(filtered_wavs.items()):
            # Load the audio in memory (results is a torch.tensor similar to a torchaudio.load("example.wav"))
            source_audio, freq = load_wav_from_scp(file)
            # Anonymize the audio
            process_wav(uttid, freq, source_audio)
            # Write to the wav.scp entry
            writer.writelines(f"{uttid} {results_dir / f'{uttid}.wav'}\n") 

    # Use logger for displaying informations
    logger.info('Done')
    ### For more advanced (batch/multiprocess anon function, checkout other implantations)
    ### Good luck!
