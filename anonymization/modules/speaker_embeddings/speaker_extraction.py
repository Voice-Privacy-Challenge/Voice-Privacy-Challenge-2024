import logging
from tqdm import tqdm
from pathlib import Path
import torch
import torchaudio
from tqdm.contrib.concurrent import process_map
import time
from torch.multiprocessing import set_start_method
from itertools import repeat, cycle
import numpy as np

from .extraction.embedding_methods import SpeechBrainVectors, StyleEmbeddings
from .extraction.ims_speaker_extraction_methods import normalize_wave
from .speaker_embeddings import SpeakerEmbeddings
from utils import read_kaldi_format

set_start_method('spawn', force=True)
logger = logging.getLogger(__name__)

class SpeakerExtraction:

    def __init__(self, devices: list, settings: dict, results_dir: Path = None, model_dir: Path = None,
                 save_intermediate=True, force_compute=False):
        self.devices = devices
        self.n_processes = len(self.devices)
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_extraction', False)

        self.emb_model_path = settings['emb_model_path']
        self.vec_type = settings['vec_type']
        self.emb_level = settings['emb_level']

        if results_dir:
            self.results_dir = results_dir
        elif 'extraction_results_path' in settings:
            self.results_dir = settings['extraction_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        model_dir = Path(model_dir or self.emb_model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f'ASV model {model_dir} does not exist!')

        # if not (model_dir / "hyperparams.yaml").exists():
        #     raise FileNotFoundError(f'ASV model {model_dir / "hyperparams.yaml"} does not exist!')

        self.model_hparams = {
            'vec_type': self.vec_type,
            'model_path': model_dir or self.emb_model_path,
        }

        self.extractors = [create_extractors(hparams=self.model_hparams, device=device) for device, process in zip(cycle(devices), range(len(devices)))]

    def extract_speakers(self, dataset_path, dataset_name=None, emb_level=None):
        dataset_name = dataset_name if dataset_name is not None else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')
        utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
        wav_scp = read_kaldi_format(dataset_path / 'wav.scp')
        spk2gender = read_kaldi_format(dataset_path / 'spk2gender')
        emb_level = emb_level if emb_level is not None else self.emb_level

        speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=self.devices[0])

        if (dataset_results_dir / 'speaker_vectors.pt').exists() and not self.force_compute:
            logger.info('No speaker extraction necessary; load existing embeddings instead...')
            speaker_embeddings.load_vectors(dataset_results_dir)
            # assume the loaded vectors are computed according to the setting in config
            speaker_embeddings.emb_level = emb_level
        else:
            logger.info(f'Extract embeddings of {len(wav_scp)} utterances')
            speaker_embeddings.new = True

            if self.n_processes > 1:
                sleeps = [10 * i for i in range(self.n_processes)]
                indices = np.array_split(np.arange(len(wav_scp)), self.n_processes)
                wav_scp_items = list(wav_scp.items())
                wav_scp_list = [dict([wav_scp_items[ind] for ind in chunk]) for chunk in indices]
                # multiprocessing
                job_params = zip(wav_scp_list, repeat(self.extractors), sleeps, self.devices,
                                 repeat(self.model_hparams), list(range(self.n_processes)))
                returns = process_map(extraction_job, job_params, max_workers=self.n_processes)
                vectors = torch.concat([x[0].to(self.devices[0]) for x in returns], dim=0)
                utts = [x[1] for x in returns]
                utts = list(np.concatenate(utts))
            else:
                vectors, utts = extraction_job([wav_scp, self.extractors[0], 0, self.devices[0], self.model_hparams, 0])
                vectors = torch.stack(vectors, dim=0)

            speakers = [utt2spk[utt] for utt in utts]
            genders = [spk2gender[speaker] for speaker in speakers]

            speaker_embeddings.set_vectors(vectors=vectors, identifiers=utts, speakers=speakers, genders=genders)

            if emb_level == 'spk':
                speaker_embeddings = speaker_embeddings.convert_to_spk_level()
            if self.save_intermediate:
                speaker_embeddings.save_vectors(dataset_results_dir)

        return speaker_embeddings


def create_extractors(hparams, device):
    extractors = []
    for single_vec_type in hparams['vec_type'].split('+'):
        if single_vec_type in {'xvector', 'ecapa'}:
            extractors.append(SpeechBrainVectors(vec_type=single_vec_type, model_path=Path(hparams['model_path']),
                                                 device=device))
        elif single_vec_type == 'style-embed':
            extractors.append(StyleEmbeddings(model_path=Path(hparams['model_path']), device=device))
        else:
            raise ValueError(f'Invalid vector type {single_vec_type}!')

    return extractors


def extraction_job(data):
    wav_scp, speaker_extractors, sleep, device, model_hparams, job_id = data
    time.sleep(sleep)

    if speaker_extractors is None:
        speaker_extractors = create_extractors(hparams=model_hparams, device=device)

    vectors = []
    utts = []
    for utt, wav_path in tqdm(wav_scp.items(), desc=f'Job {job_id}', leave=True):
        if isinstance(wav_path, list):
            wav_path = wav_path[1]
        signal, fs = torchaudio.load(wav_path)
        norm_wave = normalize_wave(signal, fs, device=device)

        try:
            spk_embs = [extractor.extract_vector(audio=norm_wave, sr=fs) for extractor in speaker_extractors]
        except RuntimeError as e:
            logger.warn(f'Runtime error: {utt}, {signal.shape}, {norm_wave.shape}')
            continue

        if len(spk_embs) == 1:
            vector = spk_embs[0]
        else:
            vector = torch.cat(spk_embs, dim=0)
        vectors.append(vector)
        utts.append(utt)

    return vectors, utts
