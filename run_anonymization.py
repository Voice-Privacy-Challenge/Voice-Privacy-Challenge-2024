from pathlib import Path
from argparse import ArgumentParser
import torch
import subprocess
import sys

from utils import parse_yaml, get_datasets, check_dependencies, setup_logger

logger = setup_logger(__name__)

def shell_run(cmd):
    if subprocess.run(['bash', cmd]).returncode != 0:
        logger.error(f'Failed to bash execute: {cmd}')
        sys.exit(1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='anon_config.yaml')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--force_compute', default=False, type=bool)
    args = parser.parse_args()

    config = parse_yaml(Path(args.config))
    datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    if config['pipeline'] == "mcadams":
        from anonymization.pipelines.mcadams import McAdamsPipeline as pipeline
    elif config['pipeline'] == "sttts":
        shell_run('anonymization/pipelines/sttts/install.sh')
        check_dependencies('anonymization/pipelines/sttts/requirements.txt')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts/download_precomputed_intermediate_repr.sh')
        from anonymization.pipelines.sttts import STTTSPipeline as pipeline
    elif config['pipeline'] == "nac":
        shell_run('anonymization/pipelines/nac/install.sh')
        if devices[0] == torch.device('cpu'):
            from anonymization.pipelines.nac.nac_pipeline import NACPipeline as pipeline
        else:
            from anonymization.pipelines.nac.nac_pipeline_accelerate import NACPipeline as pipeline
    elif config['pipeline'] == "asrbn":
        shell_run('anonymization/pipelines/asrbn/install.sh')
        check_dependencies('anonymization/pipelines/asrbn/requirements.txt')
        from anonymization.pipelines.asrbn import ASRBNPipeline as pipeline
    elif config['pipeline'] == "template":
        from anonymization.pipelines.template import TemplatePipeline as pipeline
    else:
        raise ValueError(f"Pipeline {config['pipeline']} not defined/imported")

    logger.info(f'Running pipeline: {config["pipeline"]}')
    p = pipeline(config=config, force_compute=args.force_compute, devices=devices)
    p.run_anonymization_pipeline(datasets)
