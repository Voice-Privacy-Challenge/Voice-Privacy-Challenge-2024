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
    def __init__(self, config):
        self.config = config
        self.libri_360_data_dir = Path(config['dataset_libri_360']) \
            if 'dataset_libri_360' in config else None
        self.modules_config = config['modules']
        return

    def run_anonymization_pipeline(self, datasets):
        # anonymize each dataset
        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            print(f'{i + 1}/{len(datasets)}: Processing {dataset_name}...')
            process_data(dataset_path=dataset_path,
                         anon_level=self.modules_config['anon_level'],
                         results_dir=self.config['results_dir'],
                         settings=self.modules_config)
            print('Done')

        # anonymize libri360 if it exists
        if self.libri_360_data_dir:
            print(f'Processing libri_train_360...')
            process_data(dataset_path=self.libri_360_data_dir,
                    anon_level=self.modules_config['anon_level_libri_360'],
                    results_dir=self.config['results_dir'],
                    settings=self.modules_config)

            print('Done')
        return

if __name__ == "__main__":
    print(__doc__)
