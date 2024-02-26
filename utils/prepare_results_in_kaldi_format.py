from pathlib import Path
from collections import defaultdict
from shutil import copy
from utils import save_kaldi_format, create_clean_dir, read_kaldi_format, get_datasets
from .logger import setup_logger
import os,sys

logger = setup_logger(__name__)


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


def list_wav_files_recursively(folder_path):
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


def check_files(ori_folder, anon_folder, required_files):
    skip=True
    for required_file in required_files:
        if not os.path.exists(anon_folder / required_file):
            return False   
            
        if 'wav.scp' in str(required_file):
            wav_files = list_wav_files_recursively(anon_folder)
            if len(wav_files) == 0:
                    logger.error(f"Directory {anon_folder} doen't have any audios.")
                    exit()
            
            lines = open(anon_folder / required_file).readlines()
            if len(lines) == 0:
                return False
            for line in lines:
                    skip = line.strip().split(' ')[-1] in wav_files
                    if not skip:
                        return False
        else:
            with open(ori_folder / required_file, 'rb') as f1, open(anon_folder / required_file, 'rb') as f2:
                content1 = f1.read()
                content2 = f2.read()
                if content1 == content2:
                    skip=True
    return skip


def create_kaldi_formart_data(ori_folder, anon_folder, required_files):
    logger.info(f"Create Kaldi format files for {anon_folder}")
    for file in required_files:
        ori_file = ori_folder / file
        anon_file = anon_folder / file
        copy(ori_file, anon_file)
        if file == 'wav.scp':
            with open(anon_file, 'w') as fp:
                wav_files = list_wav_files_recursively(anon_folder)
                dirname = Path(wav_files[0]).parent
                for line in open(ori_file):
                    temp = line.strip().split(' ')
                    token = temp[0]
                    audio_path = dirname / token
                    fp.write(f"{token} {audio_path}.wav\n")

def check_kaldi_formart_data(config): 
    logger.info('Check Kaldi format files')
    # 1) check datastes exist: anonymized dev$suffix test$suffix and anonymized train-clean-360$suffix
    dataset_dict = get_datasets(config)
    output_path = config['data_dir']
    suffix = config['anon_data_suffix']
    
    if 'train_data_name' in config:
        # in conf/eval_post.yaml, train_data_name = train-clean-360$suffix, ori_train_data_name=train-clean-360
        ori_train_data_name = config['train_data_name'].split(suffix)[0]
        dataset_dict[ori_train_data_name] = Path(config['data_dir'], ori_train_data_name)
    
    anon_folders = [folder for folder in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, folder)) and folder.endswith(suffix) and 'asr' not in folder]
    for dataset, orig_dataset_path in dataset_dict.items():
        out_data_split = output_path / f'{dataset}{suffix}'
        if not os.path.exists(out_data_split):
            logger.error(f"Directory {out_data_split} does not exist. Please prepare your anonymized audio in a correct format.")
            exit()
    # 2) check files in datasets exits and correct            
    for anon_folder in anon_folders:
        anon_folder = output_path / anon_folder
        ori_folder =  output_path / os.path.basename(anon_folder).split(suffix)[0]
        required_files = [file for file in os.listdir(ori_folder) if os.path.isfile(os.path.join(ori_folder, file))]
        skip = check_files(ori_folder, anon_folder,required_files)
    
        # 3) create kaldi format data
        if not skip:
            create_kaldi_formart_data(ori_folder, anon_folder, required_files)
        else:
            logger.info(f'Kaldi format files in {anon_folder} are correct.')
