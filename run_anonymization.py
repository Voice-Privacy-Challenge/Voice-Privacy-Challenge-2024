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
    args = parser.parse_args()

    config = parse_yaml(Path('configs', args.config))
    datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    logger = setup_logger(__name__)
    if config['pipeline'] == "mcadams":
        from anonymization.pipelines.mcadams import McAdamsPipeline as pipeline
    elif config['pipeline'] == "sttts":
        subprocess.run(['bash', 'anonymization/pipelines/sttts/sttts_install.sh'])
        check_dependencies('anonymization/pipelines//sttts/sttts_requirements.txt')
        from anonymization.pipelines.sttts import STTTSPipeline as pipeline
    elif config['pipeline'] == "nac":
        subprocess.run(['bash', 'anonymization/modules/nac/install_nac.sh'])
        from anonymization.pipelines.nac.nac_pipeline import NACPipeline as pipeline
    else:
        raise ValueError(f"Pipeline {config['pipeline']} not defined/imported")

    logger.info(f'Running pipeline: {config["pipeline"]}')
    p = pipeline(config=config, force_compute=args.force_compute, devices=devices)
    p.run_anonymization_pipeline(datasets)
