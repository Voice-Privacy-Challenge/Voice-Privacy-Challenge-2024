from ...pipelines.pipeline import  Pipeline
from ...modules.nac.anonymizer import Anonymizer as NACanonymizer

import os

class NACPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Initialize NAC model.

        Args:
            config: only really holds optional path where the checkpoint is found.
                Setting it to None will just load from the default checkpoint, which is what you want to do 90% of the
                time anyway. Trust me bro.
            force_compute: unused, it's just there for API consistency for now.
            devices: for now, we only support single-GPU inference, so only the first element of the list will be used.
                If you really want to make your GPUs go brrrr, consider checking out the original repo of the system.
        """
        self.device = devices[0]  # completely ignoring what comes afterward
        checkpoint_dir = os.path.expanduser(config['model']['checkpoint_dir'])
        voice_dirs = [config['model']['voice_dir']]  # only one voice dir for now

        self.anonymizer = NACanonymizer(
            checkpoint_dir=checkpoint_dir,
            voice_dirs=voice_dirs
        ).to(self.device)


    def run_anonymization_pipeline(self, datasets):
        print('Anonymization process todo.')