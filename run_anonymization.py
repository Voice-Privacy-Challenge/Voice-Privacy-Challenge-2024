import logging
from pathlib import Path
from argparse import ArgumentParser
import torch

from anonymization.pipelines.sttts_pipeline import STTTSPipeline
from utils import parse_yaml, get_datasets, check_dependencies

PIPELINES = {
    'sttts': STTTSPipeline
}

if __name__ == '__main__':
    check_dependencies('requirements.txt')
    parser = ArgumentParser()
    parser.add_argument('--config', default='anon_config.yaml')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--force_compute', default=False, type=bool)
    args = parser.parse_args()

    config = parse_yaml(Path('configs', args.config))
    datasets = get_datasets(config)
    #datasets = {'train-clean-360': Path(config['data_dir'], 'train-clean-360')}

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    with torch.no_grad():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s- %(levelname)s - %(message)s')
        logging.info(f'Running pipeline: {config["pipeline"]}')
        pipeline = PIPELINES[config['pipeline']](config=config, force_compute=args.force_compute, devices=devices)
        pipeline.run_anonymization_pipeline(datasets)
