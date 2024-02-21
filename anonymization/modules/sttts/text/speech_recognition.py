from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import time
from torch.multiprocessing import set_start_method
from itertools import cycle, repeat
import numpy as np
from pathlib import Path

from .text import Text
from .recognition.ims_asr import ImsASR
from utils import read_kaldi_format, setup_logger

set_start_method('spawn', force=True)
logger = setup_logger(__name__)

class SpeechRecognition:

    def __init__(self, devices, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.devices = devices
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_recognition', False)
        self.n_processes = len(self.devices)

        self.model_hparams = settings

        if results_dir:
            self.results_dir = results_dir
        elif 'results_path' in settings:
            self.results_dir = settings['results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        self.asr_models = [create_model_instance(hparams=self.model_hparams, device=device) for device, process in zip(cycle(devices), range(len(devices)))]
        self.is_phones = (self.asr_models[0].output == 'phones')

    def recognize_speech(self, dataset_path, dataset_name=None, utterance_list=None):
        dataset_name = dataset_name if dataset_name else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')

        utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
        texts = Text(is_phones=self.is_phones)

        if (dataset_results_dir / 'text').exists() and not self.force_compute:
            # if the text created from this ASR model already exists for this dataset and a computation is not
            # forced, simply load the text
            texts.load_text(in_dir=dataset_results_dir)

        if len(texts) == len(utt2spk):
            logger.info('No speech recognition necessary; load existing text instead...')
        else:
            if len(texts) > 0:
                logger.info(f'No speech recognition necessary for {len(texts)} of {len(utt2spk)} utterances')
            # otherwise, recognize the speech
            dataset_results_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f'Recognize speech of {len(utt2spk)} utterances...')
            wav_scp = read_kaldi_format(dataset_path / 'wav.scp')

            utterances = []
            for utt, spk in utt2spk.items():
                if utt in texts.utterances:
                    continue
                if utterance_list and utt not in utterance_list:
                    continue
                if utt in wav_scp:
                    utterances.append((utt, spk, wav_scp[utt]))

            save_intermediate = self.save_intermediate and not utterance_list
            start = time.time()

            if self.n_processes == 1:
                new_texts = [recognition_job([utterances, self.asr_models[0],
                                             dataset_results_dir, 0, self.devices[0], self.model_hparams, None,
                                             save_intermediate])]
            else:
                sleeps = [10 * i for i in range(self.n_processes)]
                indices = np.array_split(np.arange(len(utterances)), self.n_processes)
                utterance_jobs = [[utterances[ind] for ind in chunk] for chunk in indices]
                # multiprocessing
                job_params = zip(utterance_jobs, repeat(self.asr_models), repeat(dataset_results_dir), sleeps,
                                 self.devices, repeat(self.model_hparams), list(range(self.n_processes)),
                                 repeat(save_intermediate))
                new_texts = process_map(recognition_job, job_params, max_workers=self.n_processes)

            end = time.time()
            total_time = round(end - start, 2)
            logger.info(f'Total time for speech recognition: {total_time} seconds ({round(total_time / 60, 2)} minutes / '
                  f'{round(total_time / 60 / 60, 2)} hours)')
            texts = self._combine_texts(main_text_instance=texts, additional_text_instances=new_texts)

            if save_intermediate:
                texts.save_text(out_dir=dataset_results_dir)
                self._remove_temp_files(out_dir=dataset_results_dir)

        return texts

    def _combine_texts(self, main_text_instance, additional_text_instances):
        for add_text_instance in additional_text_instances:
            main_text_instance.add_instances(sentences=add_text_instance.sentences,
                                             utterances=add_text_instance.utterances,
                                             speakers=add_text_instance.speakers)

        return main_text_instance

    def _remove_temp_files(self, out_dir):
        temp_text_files = [filename for filename in out_dir.glob('text*') if filename.name != 'text']
        temp_utt2spk_files = [filename for filename in out_dir.glob('utt2spk*') if filename.name != 'utt2spk']

        for file in temp_text_files + temp_utt2spk_files:
            file.unlink()


def create_model_instance(hparams, device):
    recognizer = hparams.get('recognizer')
    if recognizer == 'ims':
        return ImsASR(**hparams, device=device)
    else:
        raise ValueError(f'Invalid recognizer option: {recognizer}')


def recognition_job(data):
    utterances, asr_model, out_dir, sleep, device, model_hparams, job_id, save_intermediate = data
    time.sleep(sleep)

    add_suffix = f'_{job_id}' if job_id is not None else None
    job_id = job_id or 0

    texts = Text(is_phones=(asr_model.output == 'phones'))
    i = 0
    for utt, spk, wav_path in tqdm(utterances, desc=f'Job {job_id}', leave=True):
        sentence = asr_model.recognize_speech_of_audio(audio_file=wav_path)
        texts.add_instance(sentence=sentence, utterance=utt, speaker=spk)

        i += 1
        if i % 100 == 0 and save_intermediate:
            texts.save_text(out_dir=out_dir, add_suffix=add_suffix)

    if save_intermediate:
        texts.save_text(out_dir=out_dir, add_suffix=add_suffix)

    return texts
