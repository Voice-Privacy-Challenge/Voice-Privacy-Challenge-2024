#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-

from pathlib import Path

from ...modules.asrbn.anonymise_dir import process_data

from .. import Pipeline, get_anon_level_from_config
from utils import setup_logger

logger = setup_logger(__name__)

class ASRBNPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Instantiates a ASRBNVQPipeline.

        This pipeline consists of:
                  ->      F0 (yaapt: no transformation)    --\
            input ->    ASR-BN (vq transformed or not)  --->  Speech synthesis (hifigan)
                                                                          ^
                                                                          |
                                                                  Target speaker (one-hot)

        Args:
            config (dict): a configuration dictionary, e.g., see anon_ims_sttts_pc.yaml
            force_compute (bool): if True, forces re-computation of
                all steps. otherwise uses saved results.
            devices (list): a list of torch-interpretable devices
        """
        self.config = config
        self.gpu_devices = devices
        self.force_compute = force_compute
        self.modules_config = config['modules']

    def run_anonymization_pipeline(self, datasets):
        # anonymize each dataset
        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            anon_level = get_anon_level_from_config(self.modules_config, dataset_name)
            print(f'{i + 1}/{len(datasets)}: ASR-BN processing of "{dataset_name}" at anon_level "{anon_level}"...')
            process_data(dataset_path=dataset_path,
                         anon_level=anon_level,
                         results_dir=self.config['results_dir'],
                         settings=self.modules_config,
                         force_compute=self.force_compute,
                         )
