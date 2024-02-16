# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd
from typing import List
import multiprocessing
parser = ArgumentParser()
parser.add_argument('--config', default='config_eval.yaml')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()
logger = logging.getLogger(__name__)
from datetime import datetime

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch
import time
import shutil
import itertools

from evaluation import evaluate_asv, train_asv_eval, evaluate_asr, train_asr_eval, evaluate_ser
from utils import (parse_yaml, scan_checkpoint, combine_asr_data, get_datasets,
                   save_yaml)

def get_evaluation_steps(params):
    eval_steps = {}
    if 'privacy' in params:
        eval_steps['privacy'] = list(params['privacy'].keys())
    if 'utility' in params:
        eval_steps['utility'] = list(params['utility'].keys())

    if 'eval_steps' in params:
        param_eval_steps = params['eval_steps']

        for eval_part, eval_metrics in param_eval_steps.items():
            if eval_part not in eval_steps:
                raise KeyError(f'Unknown evaluation step {eval_part}, please specify in config')
            for metric in eval_metrics:
                if metric not in eval_steps[eval_part]:
                    raise KeyError(f'Unknown metric {metric}, please specify in config')
        return param_eval_steps
    return eval_steps


def save_result_summary(out_dir, results_dict, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out_dir / 'config.yaml')

    with open(out_dir / 'results.txt', 'w') as f:
        f.write(f'---- Time: {datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")} ----\n')
        if 'ser' in results_dict:
            f.write('\n')
            f.write('---- SER results ----\n')
            f.write(results_dict['ser'].sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')
        if 'asv' in results_dict:
            f.write('\n')
            f.write('---- ASV results ----\n')
            f.write(results_dict['asv'].sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')
        if 'asr' in results_dict:
            f.write('\n')
            f.write('---- ASR results ----\n')
            f.write(results_dict['asr'].sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')


if __name__ == '__main__':
    multiprocessing.set_start_method("fork",force=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s- %(levelname)s - %(message)s')

    params = parse_yaml(Path('configs', args.config))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_data_dir = params['data_dir']
    anon_suffix = params['anon_data_suffix']
    eval_steps = get_evaluation_steps(params)

    results = {}

    # make sure given paths exist
    assert eval_data_dir.exists(), f'{eval_data_dir} does not exist'

    if 'privacy' in eval_steps:
        if 'asv' in eval_steps['privacy']:
            asv_params = params['privacy']['asv']
            if 'training' in asv_params:
                model_dir = params['privacy']['asv']['training']['model_dir']
                asv_train_params = asv_params['training']
                if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    logging.info('Perform ASV training')
                    train_asv_eval(train_params=asv_train_params, output_dir=model_dir)
                    logging.info("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
                    model_dir = scan_checkpoint(model_dir, 'CKPT')
                    shutil.copy(asv_train_params['train_config'], model_dir)

            if 'evaluation' in asv_params:
                logging.info('Perform ASV evaluation')
                model_dir = params['privacy']['asv']['evaluation']['model_dir']
                model_dir = scan_checkpoint(model_dir, 'CKPT')
                start_time = time.time()
                eval_data_name = params['privacy']['asv']['dataset_name']
                eval_pairs = []
                for name in eval_data_name:
                    for d in params['datasets']:
                        if d['name'] != name:
                            continue
                        if 'enrolls' not in d or 'trials' not in d:
                            raise ValueError(f"{name} is missing an ASV enrolls/trials split")
                    eval_pairs.extend([(f'{d["data"]}{enroll}',
                                        f'{d["data"]}{trial}')
                                       for enroll, trial in itertools.product(d['enrolls'], d['trials'])])
                asv_results = evaluate_asv(eval_datasets=eval_pairs, eval_data_dir=eval_data_dir,
                                           params=asv_params, device=device,  model_dir=model_dir,
                                           anon_data_suffix=anon_suffix)
                results['asv'] = asv_results
                logging.info("--- EER computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if 'utility' in eval_steps:
        if 'ser' in eval_steps['utility']:
            print('Perform SER evaluation')
            eval_data_name = params['utility']['ser']['dataset_name']
            ser_eval_params = params['utility']['ser']['evaluation']
            models_path = ser_eval_params['model_dir']

            eval_ser = []
            for name in eval_data_name:
                for d in params['datasets']:
                    if d['name'] != name:
                        continue
                    eval_ser.append(d['data'])
            start_time = time.time()
            ser_results = evaluate_ser(eval_ser, eval_data_dir, models_path, anon_data_suffix=anon_suffix, params=ser_eval_params, device=device)
            results['ser'] = ser_results
            print("--- SER evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

        if 'asr' in eval_steps['utility']:
            asr_params = params['utility']['asr']

            model_name = asr_params['model_name']
            backend = asr_params.get('backend', 'speechbrain').lower()

            if 'evaluation' in asr_params:
                asr_eval_params = asr_params['evaluation']
                model_path = asr_eval_params['model_dir']
                asr_model_path = scan_checkpoint(model_path, 'CKPT') or model_path

                if not model_path.exists():
                    raise FileNotFoundError(f'ASR model {model_path} does not exist!')

                start_time = time.time()
                print('Perform ASR evaluation')
                eval_data_name = params['utility']['asr']['dataset_name']
                eval_asr = []
                for name in eval_data_name:
                    for d in params['datasets']:
                        if d['name'] != name:
                            continue
                        for suff in  ["", anon_suffix]:
                            splits_to_combine = []
                            if 'enrolls' in d:
                                splits_to_combine += list(map(lambda x: str(params['data_dir'])+"/"+d['data']+x+suff, d['enrolls']))
                                combine_asr_data(splits_to_combine, str(params['data_dir'])+"/"+d['name']+"_asr"+suff)
                            if 'trials' in d:
                                splits_to_combine += list(map(lambda x: str(params['data_dir'])+"/"+d['data']+x+suff, d['trials']))
                                combine_asr_data(splits_to_combine, str(params['data_dir'])+"/"+d['name']+"_asr"+suff)

                        if 'trials' not in d and 'enrolls' not in d:
                            eval_asr.append(d['data'])
                            print("A", eval_asr)
                        else:
                            eval_asr.append(d['name']+"_asr")

                asr_results = evaluate_asr(eval_datasets=eval_asr, eval_data_dir=eval_data_dir,
                                           params=asr_eval_params, model_path=asr_model_path,
                                           anon_data_suffix=anon_suffix, device=device, backend=backend)
                results['asr'] = asr_results
                print("--- ASR evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

    if results:
        now = datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")
        results_summary_dir = params.get('results_summary_dir', Path('exp', 'results_summary', now))
        save_result_summary(out_dir=results_summary_dir, results_dict=results, config=params)
