import numpy as np
import torch


class Prosody:

    def __init__(self):
        self.utterances = {}
        self.idx2utt = {}
        self.durations = []
        self.pitches = []
        self.energies = []

        self.start_silences = []
        self.end_silences = []

        self.new = True

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        if set(self.start_silences) != {None}:
            for i in range(len(self)):
                yield self.idx2utt[i], self.durations[i], self.pitches[i], self.energies[i], self.start_silences[i], \
                      self.end_silences[i]
        else:
            for i in range(len(self)):
                yield self.idx2utt[i], self.durations[i], self.pitches[i], self.energies[i]

    def add_instance(self, utterance, duration, pitch, energy, start_silence=None, end_silence=None):
        idx = len(self)
        self.utterances[utterance] = idx
        self.idx2utt[idx] = utterance
        self.durations.append(duration)
        self.pitches.append(pitch)
        self.energies.append(energy)
        self.start_silences.append(start_silence)
        self.end_silences.append(end_silence)

    def get_instance(self, utterance):
        idx = self.utterances[utterance]
        return_dict = {
            'duration': self.durations[idx],
            'pitch': self.pitches[idx],
            'energy': self.energies[idx]
        }

        if len(self.start_silences) > 0:
            return_dict['start_silence'] = self.start_silences[idx]
            return_dict['end_silence'] = self.end_silences[idx]

        return return_dict

    def update_instance(self, utterance, duration, pitch, energy, start_silence=None, end_silence=None):
        idx = self.utterances[utterance]
        self.durations[idx] = duration
        self.pitches[idx] = pitch
        self.energies[idx] = energy
        self.start_silences[idx] = start_silence
        self.end_silences[idx] = end_silence

    def shuffle(self):
        shuffled_utterances = {}
        shuffled_durations = []
        shuffled_pitches = []
        shuffled_energies = []
        shuffled_start_silences = []
        shuffled_end_silences = []

        i = 0
        for idx in np.random.permutation(len(self)):
            shuffled_utterances[self.idx2utt[idx]] = i
            shuffled_durations.append(self.durations[idx])
            shuffled_pitches.append(self.pitches[idx])
            shuffled_energies.append(self.energies[idx])
            shuffled_start_silences.append(self.start_silences[idx])
            shuffled_end_silences.append(self.end_silences[idx])
            i += 1

        self.utterances = shuffled_utterances
        self.durations = shuffled_durations
        self.pitches = shuffled_pitches
        self.energies = shuffled_energies
        self.start_silences = shuffled_start_silences
        self.end_silences = shuffled_end_silences
        self.idx2utt = {idx: utt for utt, idx in self.utterances.items()}

    def save_prosody(self, out_dir):
        out_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.durations, out_dir / 'duration.pt')
        torch.save(self.pitches, out_dir / 'pitch.pt')
        torch.save(self.energies, out_dir / 'energy.pt')
        torch.save(self.start_silences, out_dir / 'start_silence.pt')
        torch.save(self.end_silences, out_dir / 'end_silence.pt')

        with open(out_dir / 'utterances', 'w') as f:
            for utt, _ in sorted(self.utterances.items(), key=lambda x: x[1]):
                f.write(f'{utt}\n')

    def load_prosody(self, in_dir):
        self.new = False

        self.durations = torch.load(in_dir / 'duration.pt', map_location='cpu')
        self.pitches = torch.load(in_dir / 'pitch.pt', map_location='cpu')
        self.energies = torch.load(in_dir / 'energy.pt', map_location='cpu')
        self.start_silences = torch.load(in_dir / 'start_silence.pt', map_location='cpu')
        self.end_silences = torch.load(in_dir / 'end_silence.pt', map_location='cpu')

        self.utterances = {}
        i = 0
        with open(in_dir / 'utterances', 'r') as f:
            for line in f:
                self.utterances[line.strip()] = i
                i += 1
        self.idx2utt = {idx: utt for utt, idx in self.utterances.items()}
