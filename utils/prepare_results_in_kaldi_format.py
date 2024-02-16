from shutil import copy
from pathlib import Path
import torchaudio
import os
from collections import defaultdict

from utils import save_kaldi_format, create_clean_dir, read_kaldi_format


def transform_spk2gender(session_spk2gender, session_utt2spk, global_utt2spk):
    utt2gender = {utt: session_spk2gender[spk] for utt, spk in session_utt2spk.items()}
    global_spk2gender = {spk: utt2gender[utt] for utt, spk in global_utt2spk.items()}
    return global_spk2gender


def combine_asr_data(input_dirs, output_dir):
    output_dir = Path(output_dir)
    create_clean_dir(output_dir)

    files = ['text', 'utt2spk', 'spk2gender', 'wav.scp']

    for file in files:
        data = {}
        for in_dir in input_dirs:
            in_dir = Path(in_dir)
            data.update(read_kaldi_format(in_dir / file))
        save_kaldi_format(data, output_dir / file)

    spk2utt = defaultdict(list)
    for in_dir in input_dirs:
        in_dir = Path(in_dir)
        for spk, utt_list in read_kaldi_format(in_dir / 'spk2utt').items():
            spk2utt[spk].extend(utt_list)
    spk2utt = {spk: sorted(utt_list) for spk, utt_list in spk2utt.items()}
    save_kaldi_format(spk2utt, output_dir / 'spk2utt')
