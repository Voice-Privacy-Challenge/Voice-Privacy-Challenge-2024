from pathlib import Path
from collections import defaultdict
from shutil import copy
from utils import save_kaldi_format, create_clean_dir, read_kaldi_format, get_datasets
import os

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

def check_file(ori_file, anon_file):
    skip=True
    if 'wav.scp' in str(anon_file):
        for line in open(anon_file):
            if 'anon_wav' in line.strip():
                skip += os.path.exists(line.strip().split(' ')[-1])
            else:
                return False
    else:
        with open(ori_file, 'rb') as f1, open(anon_file, 'rb') as f2:
            content1 = f1.read()
            content2 = f2.read()
            if content1 == content2:
                skip=True
    return skip


def check_kaldi_formart_data(config): 
    dataset_dict = get_datasets(config)
    output_path = config['data_dir']
    suffix = config['anon_data_suffix']

    for dataset, orig_dataset_path in dataset_dict.items():
        # do the transformation for original and anonymized versions of each dataset
        # if it is the original data, the information is simply copied from an external source of the data
        # if it is the anonymized data, some information (e.g. the wav.scp) is generated to match the information of the anonymized data
        
        out_data_split = output_path / f'{dataset}{suffix}'
        #out_data_split.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(out_data_split):
            print(f"Directory {out_data_split} does not exist. Please prepare your anoymized audio in a correct formart.")
            exit()
        
        # these files are never changed during anonymization and always copied from the original
        copy_files = ['spk2utt', 'text', 'utt2spk','spk2gender','utt2dur', "wav.scp"]

        if 'trials' in dataset:
            # the trials file only exists for the trials subsets
            copy_files += ['trials']
        elif 'enrolls' in dataset:
            # the enrolls file only exists for the enrolls subsets
            copy_files += ['enrolls']

        em_copy_files = ['spk2fold', 'spk2gender','spk2utt',  'text', 'text_no_norm','utt2emo','utt2spk', "wav.scp"]

        if "IEMOCAP" in dataset:
            copy_files = em_copy_files

        for file in copy_files:
            ori_file = orig_dataset_path / file
            anon_file = out_data_split / file
            skip = False
            
            # Check if  anonymized file already exists and check
            if os.path.exists(anon_file):
                skip = check_file(ori_file, anon_file)
            
            # Create anonymized file if not exits or not correct
            if not skip:
                copy(ori_file, anon_file)
                if file=='wav.scp':
                    fp = open(anon_file,'w')
                    for line in open(ori_file):
                        temp = line.strip().split(' ')
                        token = temp[0]
                        audio_path = out_data_split / 'anon_wav' / token
                        fp.write("%s %s.wav\n"%(token, audio_path))
                    fp.close()
