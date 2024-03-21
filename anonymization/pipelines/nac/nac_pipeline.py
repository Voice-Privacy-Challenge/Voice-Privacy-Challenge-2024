from ...pipelines.pipeline import  Pipeline
from ...modules.nac.anonymizer import Anonymizer as NACanonymizer
from .data import SCPPathDataset
from utils import setup_logger

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
            devices: for now, we only support single-GPU inference, so only the first element of the list will be used.
                If you really want to make your GPUs go brrrr, consider checking out the original repo of the system.
        """
        self.device = devices[0]  # completely ignoring what comes afterward
        checkpoint_dir = os.path.expanduser(config['modules']['model']['checkpoint_dir'])
        voice_dirs = [config['modules']['model']['voice_dir']]  # only one voice dir for now

        logger.info('Instantiating anonymizer model...')
        self.anonymizer = NACanonymizer(
            checkpoint_dir=checkpoint_dir,
            voice_dirs=voice_dirs
        ).to(self.device)
        logger.info('Done.')

        # keep the config-ball (i literally just made up that name) for later
        self.config = config


    def run_anonymization_pipeline(self, datasets):
        # create result folder if needed
        results_dir = self.config['results_dir']
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name}...")

            # prepare scp dataset
            scp_dataset = SCPPathDataset(
                scp_file=os.path.join(dataset_path, "wav.scp"),
                root='.',  # this assumes that the method is always called by the main script
                ds_type='libri',
                voice_folder=None,  # we are using predefined mappings so this doesnt matter
            )

            # create individual result folder for dataset
            ds_out_folder = os.path.join(results_dir, dataset_name)
            if not os.path.isdir(ds_out_folder):
                os.mkdir(ds_out_folder)

            # barbarically loop through dataset, anonymize, write to disk
            # could be sped up by using multiple jobs or sharing across gpus
            # this implementation is more basic, see original repo to run across multiple gpus with Accelerate
            tot_files = len(scp_dataset)
            for i, (utt_id, file_path, basename, proxy_speaker) in enumerate(scp_dataset, start=1):
                logger.info(f'[{i}/{tot_files}] Anonymizing {utt_id}\t| proxy spk {proxy_speaker}')

                anon_wav = self.anonymizer(file_path, target_voice_id=proxy_speaker)
                # encodec operates at 24k, needs resampling
                anon_wav_16k = resample(anon_wav, orig_sr=self.anonymizer.model.config.sample_rate, target_sr=16000)

                out_path = os.path.join(ds_out_folder, basename)
                logger.info(f'Saving to {out_path}')
                sf.write(out_path, anon_wav_16k, samplerate=16000)

            logger.info('Done.')
