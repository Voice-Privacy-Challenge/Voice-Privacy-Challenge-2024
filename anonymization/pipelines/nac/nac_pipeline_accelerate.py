from ...pipelines.pipeline import Pipeline
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

            # the mapping file does not have _female or _male (for gender equality or for laziness)
            mapping_file_name = f'speaker_mapping_{dataset_name.replace("_f", "").replace("_m", "")}.json'


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

            new_env = os.environ.copy()
            new_env['CUDA_VISIBLE_DEVICES'] = ','.join([str(dv.index) for dv in self.devices])

            if '360' not in dataset_name:
                speaker_mappings_root = self.config['modules']['speaker_mappings_root']
                args_to_run.extend(('--mapping_file', os.path.join(
                    speaker_mappings_root, mapping_file_name)))
                # otherwise, train 360 is anonymized utterance level

            subprocess.run(args_to_run, env=new_env)
            logger.info('Done.')