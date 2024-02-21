from pathlib import Path

from .anonymization import *
from .prosody import Prosody


class ProsodyAnonymization:

    def __init__(self, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_anonymization', False)
        anonymizer_type = settings.get('anonymizer_type', 'ims')

        if results_dir:
            self.results_dir = results_dir
        elif 'anon_results_path' in settings:
            self.results_dir = settings['anon_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        if anonymizer_type == 'ims':
            random_offset_lower = settings.get('random_offset_lower', None)
            random_offset_higher = settings.get('random_offset_higher', None)
            self.anonymization = ImsProsodyAnonymization(random_offset_lower=random_offset_lower,
                                                         random_offset_higher=random_offset_higher)

    def anonymize_prosody(self, prosody, dataset_name):
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')

        anon_prosody = Prosody()

        if (dataset_results_dir / 'utterances').exists() and not self.force_compute:
            anon_prosody.load_prosody(dataset_results_dir)
            unprocessed_utts = prosody.utterances.keys() - anon_prosody.utterances.keys()
        else:
            unprocessed_utts = prosody.utterances.keys()

        for utt in unprocessed_utts:
            prosodic_elements = prosody.get_instance(utt)
            anon_prosodic_elements = self.anonymization.anonymize_values(**prosodic_elements)
            anon_prosody.add_instance(utt, **anon_prosodic_elements)

        if unprocessed_utts and self.save_intermediate:
            anon_prosody.save_prosody(out_dir=dataset_results_dir)

        return anon_prosody