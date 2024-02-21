from pathlib import Path
from collections import defaultdict

from utils import save_kaldi_format, create_clean_dir, read_kaldi_format


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
