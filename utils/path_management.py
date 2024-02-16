from pathlib import Path
import shutil
import os
import glob
from multiprocessing import Manager

def create_clean_dir(dir_name:Path, force:bool = True):
    if dir_name.exists() and force:
        remove_contents_in_dir(dir_name)
    else:
        dir_name.mkdir(exist_ok=True, parents=True)


def copy_data_dir(dataset_path, output_path):
    # Copy utt2spk wav.scp and so on, but not the directories inside (may contains clear or anonymzied *.wav)
    os.makedirs(output_path, exist_ok=True)
    for p in glob.glob(str(dataset_path / '*'), recursive=False):
        if os.path.isfile(p):
            shutil.copy(p, output_path)


def remove_contents_in_dir(dir_name:Path):
    # solution from https://stackoverflow.com/a/56151260
    for path in dir_name.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def transform_path(file_path, parent_dir=None):
    if not file_path:
        return None
    file_path = Path(file_path)
    if parent_dir and not file_path.is_absolute():
        file_path = parent_dir / file_path
    return file_path


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*****')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return Path(sorted(cp_list)[-1])


def find_asv_model_checkpoint(model_dir):
    if list(model_dir.glob('CKPT+*')):  # select latest checkpoint
        model_dir = scan_checkpoint(model_dir, 'CKPT')
    return model_dir


def get_datasets(config):
    datasets = {}
    data_dir = config.get('data_dir', None).expanduser() # if '~' is given in path then manually expand
    for dataset in config['datasets']:
        no_sub = True
        for subset in ['trials', 'enrolls']:
            if subset in dataset:
                for subset in dataset[subset]:
                    dataset_name = f'{dataset["data"]}{subset}'
                    datasets[dataset_name] = Path(data_dir, dataset_name)
                    no_sub = False
        if no_sub:
            dataset_name = f'{dataset["data"]}'
            datasets[dataset_name] = Path(data_dir, dataset_name)
    return datasets
