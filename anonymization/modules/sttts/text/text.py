from pathlib import Path
import numpy as np

from utils import read_kaldi_format, save_kaldi_format


class Text:

    def __init__(self, is_phones=False):
        self.sentences = []
        self.utterances = []
        self.speakers = []
        self.utt2idx = {}

        self.new = True
        self.is_phones = is_phones

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        for i in range(len(self)):
            yield self.sentences[i], self.utterances[i], self.speakers[i]

    def __getitem__(self, utt):
        return self.get_instance(utt)[0]

    def add_instance(self, sentence, utterance, speaker):
        self.utt2idx[utterance] = len(self)
        self.sentences.append(sentence)
        self.utterances.append(utterance)
        self.speakers.append(speaker)

    def add_instances(self, sentences, utterances, speakers):
        if len(set(utterances) & set(self.utterances)) > 0:
            remove_indices = [i for i, utt in enumerate(utterances) if utt in self.utterances]
            for idx in reversed(remove_indices):
                del sentences[idx]
                del utterances[idx]
                del speakers[idx]

        self.utt2idx.update({utt: len(self) + i for i, utt in enumerate(utterances)})
        self.sentences.extend(sentences)
        self.utterances.extend(utterances)
        self.speakers.extend(speakers)

    def get_instance(self, utterance):
        idx = self.utt2idx[utterance]
        return self.sentences[idx], self.speakers[idx]

    def get_iterators(self, n):
        # divides the stored data into n packages and returns a list of iterators over each package
        # like __iter__, but with several iterators

        def _get_instance_by_indices(indices):
            for i in indices:
                yield self.sentences[i], self.utterances[i], self.speakers[i]

        iterator_length = len(self) // n
        it_slices = [[iterator_length * i, iterator_length * (i + 1)] if i < (n - 1)
                     else [iterator_length * i, len(self)] for i in range(n)]
        iterators = [_get_instance_by_indices(range(*it_slice)) for it_slice in it_slices]
        return iterators

    def update_instance(self, utterance, sentence):
        idx = self.utt2idx[utterance]
        self.sentences[idx] = sentence

    def remove_instances(self, utterance_list):
        remove_indices = [self.utt2idx[utt] for utt in utterance_list if utt in self.utt2idx]
        remove_indices = sorted(remove_indices, reverse=True)
        for idx in remove_indices:
            del self.sentences[idx]
            del self.utterances[idx]
            del self.speakers[idx]
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def get_instances_of_speaker(self, speaker):
        indices = [i for (i, spk) in enumerate(self.speakers) if spk == speaker]
        sentences = [self.sentences[i] for i in indices]
        utterances = [self.utterances[i] for i in indices]
        return sentences, utterances

    def shuffle(self):
        shuffled_sentences = []
        shuffled_utterances = []
        shuffled_speakers = []
        for i in np.random.permutation(len(self)):
            shuffled_sentences.append(self.sentences[i])
            shuffled_utterances.append(self.utterances[i])
            shuffled_speakers.append(self.speakers[i])
        self.sentences = shuffled_sentences
        self.utterances = shuffled_utterances
        self.speakers = shuffled_speakers
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def save_text(self, out_dir: Path, add_suffix=None):
        out_dir.mkdir(exist_ok=True, parents=True)
        add_suffix = add_suffix if add_suffix is not None else ""
        save_kaldi_format(data=list(zip(self.utterances, self.sentences)),
                          filename=out_dir / f'text{add_suffix}')
        save_kaldi_format(data=list(zip(self.utterances, self.speakers)),
                          filename=out_dir / f'utt2spk{add_suffix}')

    def load_text(self, in_dir, add_suffix=None):
        self.new = False
        add_suffix = add_suffix if add_suffix is not None else ""
        utt_1, sentences = read_kaldi_format(filename=in_dir / f'text{add_suffix}', return_as_dict=False,
                                             values_as_string=True)
        utt_2, speakers = read_kaldi_format(filename=in_dir / f'utt2spk{add_suffix}', return_as_dict=False)

        if utt_1 == utt_2:
            self.utterances = utt_1
            self.sentences = sentences
            self.speakers = speakers
        elif sorted(utt_1) == sorted(utt_2):
            self.utterances, self.sentences = zip(*sorted(zip(utt_1, sentences), key=lambda x: x[0]))
            _, self.speakers = zip(*sorted(zip(utt_2, speakers), key=lambda x: x[0]))
        else:
            raise ValueError(f'{in_dir / f"text{add_suffix}"} and {in_dir / f"utt2spk{add_suffix}"} have mismatching '
                             f'utterance keys; sentences cannot be loaded!')
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}
