from pathlib import Path
from argparse import ArgumentParser
import torch
import subprocess

from utils import parse_yaml, get_datasets, check_dependencies, setup_logger

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='anon_config.yaml')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--force_compute', default=False, type=bool)
    parser.add_argument('--anonymize_train_data', default=False, type=bool)
    args = parser.parse_args()

    config = parse_yaml(Path('configs', args.config))
    if args.anonymize_train_data:
        datasets = {'train-clean-360': Path(config['data_dir'], 'train-clean-360')}
        config['modules']['speaker_embeddings']['emb_level'] = 'utt'  # train data for eval models is anonymized on utt level
    else:
        datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    with torch.no_grad():
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s- %(levelname)s - %(message)s')
        logger = setup_logger(__name__)
        logger.info(f'Running pipeline: {config["pipeline"]}')

        if config['pipeline'] == 'sttts':
            subprocess.run(['sh', 'anonymization/pipelines/sttts_install.sh'])
            check_dependencies('ims_gan_requirements.txt')
            from anonymization.pipelines.sttts_pipeline import STTTSPipeline
            pipeline = STTTSPipeline(config=config, force_compute=args.force_compute, devices=devices)
            pipeline.run_anonymization_pipeline(datasets)
