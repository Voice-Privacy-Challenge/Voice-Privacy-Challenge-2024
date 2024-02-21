from pathlib import Path
import numpy as np
import pandas as pd
import torch
from itertools import combinations_with_replacement
from collections import defaultdict
import random

from anonymization.modules.speaker_embeddings.speaker_extraction import SpeakerExtraction
from utils import read_kaldi_format, write_table, setup_logger
from evaluation.privacy.asv.asv import ASV
from evaluation.privacy.asv.metrics.helpers import optimal_llr

logger = setup_logger(__name__)

class VoiceDistinctiveness:

    def __init__(self, spk_ext_model_dir, device, score_save_dir, plda_settings=None, distance='plda',
                 vec_type='xvector', num_per_spk='all'):
        self.num_per_spk = num_per_spk

        self.extractor = SpeakerExtraction(results_dir=score_save_dir / 'emb_xvect',
                                           devices=[device], settings={'vec_type': vec_type, 'emb_level': 'utt', 'emb_model_path': spk_ext_model_dir})

        self.asv = ASV(model_dir=spk_ext_model_dir, device=device, score_save_dir=score_save_dir, distance=distance,
                       plda_settings=plda_settings, vec_type=vec_type)

    def compute_voice_similarity_matrix(self, segments_dir_x, segments_dir_y, exp_name, out_dir):
        spk2utt_x = self._get_spk2utt(segments_dir_x)
        spk2utt_y = self._get_spk2utt(segments_dir_y)

        vectors_x = self.extractor.extract_speakers(dataset_path=segments_dir_x)
        vectors_y = self.extractor.extract_speakers(dataset_path=segments_dir_y)

        x, y = self._select_utterances(spk2utt_x, spk2utt_y)

        x_speakers, x_utts = zip(*x)
        y_speakers, y_utts = zip(*y)

        trial_speakers = []
        trial_utts = []
        for x_spk, x_utt in x:
            for y_spk, y_utt in y:
                if x_utt != y_utt:
                    trial_speakers.append([x_spk, y_spk])
                    trial_utts.append([x_utt, y_utt])

        enrol_ids = list(set(x_utts))
        enrol_vectors = torch.stack([vectors_x.get_embedding_for_identifier(utt) for utt in enrol_ids])
        test_ids = list(set(y_utts))
        test_vectors = torch.stack([vectors_y.get_embedding_for_identifier(utt) for utt in test_ids])

        segments_file = out_dir / f'segments_{exp_name}_trial.txt'
        speakers_file = out_dir / f'spk_{exp_name}_trial.txt'
        scores_file = out_dir / f'scores_output_{exp_name}'

        np.savetxt(segments_file, trial_utts, delimiter=' ', newline='\n', fmt='%s')
        np.savetxt(speakers_file, trial_speakers, delimiter=' ', newline='\n', fmt='%s')

        sim_scores, enrol_indices, test_indices = self.asv.compute_distances(enrol_vectors=enrol_vectors,
                                                                             enrol_ids=enrol_ids,
                                                                             test_vectors=test_vectors,
                                                                             test_ids=test_ids)

        scores = self.asv.compute_trial_scores(trials=trial_utts, enrol_indices=enrol_indices,
                                               test_indices=test_indices, out_file=scores_file, sim_scores=sim_scores)

        trial_speakers = pd.DataFrame(trial_speakers, columns=['enroll_spk', 'trial_spk'])
        cal_data = self._score_calibration(scores=scores, speakers=trial_speakers,
                                           out_scores_file=Path(f'{scores_file}.calibrated'),
                                           out_spk_file=Path(f'{speakers_file}.calibrated'))
        return self._compute_similarity_matrix(score_data=cal_data, out_dir=out_dir, name=exp_name)

    def gvd(self, oo_sim, pp_sim):
        return 10 * np.log10(self.diagonal_dominance(pp_sim) / self.diagonal_dominance(oo_sim))

    def deid(self, oo_sim, op_sim):
        return 1 - (self.diagonal_dominance(op_sim) / self.diagonal_dominance(oo_sim))

    def _get_spk2utt(self, segments_dir):
        spk2utt_path = Path(segments_dir, 'spk2utt')

        if spk2utt_path.exists():
            return read_kaldi_format(filename=Path(segments_dir, 'spk2utt'))
        else:
            utt2spk_path = Path(segments_dir, 'utt2spk')
            if utt2spk_path.exists():
                utt2spk = read_kaldi_format(filename=Path(segments_dir, 'utt2spk'))
                spk2utt = defaultdict(list)
                for utt, spk in utt2spk.items():
                    spk2utt[spk].append(utt)
                return spk2utt
            else:
                raise FileNotFoundError(f'Either {spk2utt_path} or {utt2spk_path} must exist!')

    def _select_utterances(self, spk2utt_x, spk2utt_y):
        if self.num_per_spk == 'all':
            x = [(spk, utt) for spk, utt_list in spk2utt_x.items() for utt in utt_list]
            y = [(spk, utt) for spk, utt_list in spk2utt_y.items() for utt in utt_list]

        else:
            logger.info("choose %d utterances for each spk to create trial" % int(self.num_per_spk))
            x = [(spk, utt) for spk, utt_list in spk2utt_x.items()
                 for utt in random.sample(utt_list, k=min(self.num_per_spk, len(utt_list)))]
            y = [(spk, utt) for spk, utt_list in spk2utt_y.items()
                 for utt in random.sample(utt_list, k=min(self.num_per_spk, len(utt_list)))]

        return x, y

    def _score_calibration(self, scores, speakers, out_scores_file, out_spk_file):
        #spk_file: spk trials
        # scores = read_table(filename=scores_file, names=['enroll_id', 'trial_id', 'score'], dtype=[str, str, float])
        # speakers = read_table(filename=spk_file, names=['enroll_spk', 'trial_spk'], dtype=str)
        data = pd.concat([speakers, scores], axis=1)
        data['target'] = (data['enroll_spk'] == data['trial_spk'])

        targets = data[data['target'] == True]['score']
        target_scores = targets.to_numpy()
        nontargets = data[data['target'] == False]['score']
        nontarget_scores = nontargets.to_numpy()

        cal_target_scores, cal_nontarget_scores = optimal_llr(target_scores, nontarget_scores, laplace=True)
        data.loc[targets.index, 'cal_score'] = cal_target_scores
        data.loc[nontargets.index, 'cal_score'] = cal_nontarget_scores
        data = data.sort_values(by=['target', 'enroll_id', 'trial_id'], ascending=[False, True, True])

        write_table(table=data[['enroll_id', 'trial_id', 'cal_score']], filename=out_scores_file)
        write_table(table=data[['enroll_spk', 'trial_spk']], filename=out_spk_file)

        return data

    def _compute_similarity_matrix(self, score_data, out_dir, name):
        # score_data: dataframe with PLDA/cosine output scores
        # name: name of the similarity matrix

        spk_list = score_data['enroll_spk'].unique().tolist()
        num_spk = len(spk_list)
        similarity_matrix = np.zeros((num_spk, num_spk))

        for i, j in combinations_with_replacement(range(num_spk), 2):
            score_data_ij = score_data[(score_data['enroll_spk'] == spk_list[i]) &
                                       (score_data['trial_spk'] == spk_list[j])]
            llr = score_data_ij['score'].to_numpy()
            c = 1 / (1 + np.exp(-(np.sum(llr) / len(llr))))
            similarity_matrix[i, j] = c  # (sum(LLR)/len(LLR))
            similarity_matrix[j, i] = c  # (sum(LLR)/len(LLR))

        np.save(f'{out_dir}/similarity_matrix_{name}', similarity_matrix)
        return similarity_matrix

    def diagonal_dominance(self, matrix):
        # Ddiag
        #matrix = np.load(matrix_file)
        n = matrix.shape[0]  # matrix dimension
        m = np.mean(matrix)  # mean of all elements
        md = np.mean(np.diag(matrix))  # mean of diagonal elements
        mnd = (n / (n - 1)) * (m - (md / n))  # mean of off-diagonal elements
        return abs(md - mnd)
