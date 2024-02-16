#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
This pipeline consists of:
                         -> non-real poles -> McAdam coef -> modified poles
    input -> LP analysis -> real poles     ---------------->                -> LP synthesis -> output
                         -> residual       ---------------->
"""

from pathlib import Path
from anonymization.modules.dsp.anonymise_dir_mcadams_rand_seed import process_data

class DSPPipeline:
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        self.config = config
        self.force_compute = force_compute
        self.modules_config = config['modules']
        return

    def run_anonymization_pipeline(self, datasets):
        # anonymize each dataset
        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            if dataset_name in self.modules_config['anon_level_spk']:
                anon_level = "spk"
            if dataset_name in self.modules_config['anon_level_utt']:
                anon_level = "utt"
            print(f'{i + 1}/{len(datasets)}: DSP processing of "{dataset_name}" at anon_level "{anon_level}" ...')
            process_data(dataset_path=dataset_path,
                         anon_level=anon_level,
                         results_dir=self.config['results_dir'],
                         settings=self.modules_config,
                         force_compute=self.force_compute,
                         )

if __name__ == "__main__":
    print(__doc__)
