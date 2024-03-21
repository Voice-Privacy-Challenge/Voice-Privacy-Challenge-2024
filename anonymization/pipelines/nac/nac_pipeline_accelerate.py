from ...pipelines.pipeline import Pipeline
from ...modules.nac.anonymizer import Anonymizer as NACAnonymizer
from utils import setup_logger
import subprocess

import os
import soundfile as sf
from librosa import resample

logger = setup_logger(__name__)


class NACPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Initialize NAC model.

        Args:
            config: dictionary of various configurations.
            force_compute: unused, it's just there for API consistency for now.
            devices: list of devices to use. They should be cuda devices.
        """
        self.devices = devices
        self.config = config

        # just instantiate the model once to download the checkpoints if needed
        NACAnonymizer(checkpoint_dir=os.path.expanduser(self.config['modules']['model']['checkpoint_dir']))



    def run_anonymization_pipeline(self, datasets):
        checkpoint_dir = os.path.expanduser(self.config['modules']['model']['checkpoint_dir'])
        voice_dir = self.config['modules']['model']['voice_dir']  # only one voice dir for now
        results_dir = self.config['data_dir']
        anon_suffix = self.config['anon_suffix']

        # create result folder if needed
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name}...")

            scp_file = os.path.join(dataset_path, 'wav.scp')
            root = '.'
            ds_type = 'libri' if 'libri' in dataset_name else 'iemocap'  # absolute abomination

            # create individual result folder for dataset
            ds_out_folder = os.path.join(results_dir, f'{dataset_name}{anon_suffix}', 'wav')
            if not os.path.isdir(ds_out_folder):
                os.makedirs(ds_out_folder)

            args_to_run = [
                'accelerate', 'launch',
                '--config_file', 'anonymization/modules/nac/accelerate_config.yaml',
                f'--num_processes={len(self.devices)}',
                'anonymization/pipelines/nac/inference.py', scp_file, ds_out_folder, checkpoint_dir,
                '--data_root', root,
                '--voice_dir', str(voice_dir),
                '--ds_type', ds_type,
                '--target_rate', '16000'
            ]
            if len(self.devices) > 1:
                args_to_run.append(f'--gpu_ids=' + ','.join([str(dv.index) for dv in self.devices]))

            if subprocess.run(args_to_run).returncode != 0:
                exit(1)
            logger.info('Done.')
