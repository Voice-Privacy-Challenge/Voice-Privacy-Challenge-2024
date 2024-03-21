from torch.utils.data import Dataset
from typing import Union
import os
import random
import json


class SCPPathDataset(Dataset):
    def __init__(
            self,
            scp_file: str,
            root: Union[str, None] = None,
            ds_type: str = "libri",
            voice_folder: Union[str, None] = None,
            speaker_level: bool = True,
            mapping_file: Union[str, None] = None
    ):
        """
        Literally just takes a Kaldi-format scp file (damn you Kaldi!) and returns utterance ids and paths.
        Optionally returns basename. Optionally appends root to the path.
        Don't use this with scp files that have fancy piping, it will not work.

        @param scp_file: path to scp file
        @param ds_type: Either 'libri' or 'vctk'.
        @param root: root to append to the paths read form scp. If None, will not be appended.
        @param speaker_level: whether anonymization should be done speaker level or utterance level
        @param voice_folder: where to take the proxy speakers. If None, the proxy speaker will always return None.
            This should be formatted as bark expects, i.e. each subfolder has the name of a proxy speaker, and inside
            is its .npz ot .wav prompt.
        @param mapping_file: path to a json file that contains speaker mappings. If given, voice_folder and
            speaker_level are ignored.
        """
        assert ds_type in ["libri", "vctk", "iemocap"], f"ds_type '{ds_type}' not allowed, must be either 'libri' or 'vctk' or 'iemocap'."

        self.ds_type = ds_type
        if self.ds_type == 'libri':
            self.spk_from_utt_id = lambda utt_id: utt_id.split('-')[0]
        else:  # this is for vctk (and now iemocap)
            self.spk_from_utt_id = lambda utt_id: utt_id.split('_')[0]
        self.voice_folder = voice_folder
        self.root = root

        # first get the available data
        with open(scp_file, 'r') as f:
            scp_lines = f.readlines()

        self.data = []
        for line in scp_lines:
            # utt_id, path = line.strip().split()
            parsed_line = line.strip().split()
            if len(parsed_line) == 2:
                utt_id, path = parsed_line
            else:
                #this is the case of train-360
                utt_id, _, _, _, _, path, _ = parsed_line
            basename = os.path.basename(path)
            if root:
                path = os.path.join(root, path)

            row = (utt_id, path, basename)
            self.data.append(row)

        # done, now with the speaker mapping
        # 1. if a mapping file is given, just use that
        if mapping_file:
            print(f"Mapping file {mapping_file} given to dataset, ignoring 'voice_folder' and 'speaker_level' parameters.")
            if os.getenv('VPC_TEST_TOOLS', 'False').lower() == "true": # For testing only!
                mapping_file = mapping_file.replace("_test_tool", "")
            with open(mapping_file) as f:
                self.speaker_mapping = json.load(f)
            self.spk_to_proxy_spk = lambda spk_id: self.speaker_mapping[spk_id]
        elif self.voice_folder:
            # 2. if a voice folder was given, use that to figure it out
            if speaker_level:
                # if anon is speaker level, speaker_mapping is a dict mapping original_spk -> proxy spk
                self.speaker_mapping = self.generate_speaker_mapping(self.voice_folder)
                self.spk_to_proxy_spk = lambda spk_id: self.speaker_mapping[spk_id]
            else:
                # if anon is utt level, speaker_mapping is just a list of proxy speakers which we randomly sample from
                # (notice that the lambda below completely ignores the spk_id input)
                self.speaker_mapping = os.listdir(self.voice_folder)
                self.spk_to_proxy_spk = lambda spk_id: random.choice(self.speaker_mapping)
        else:
            # if neither a mapping or a voice folder are given, just return fucking nothing
            self.speaker_mapping = None
            self.spk_to_proxy_spk = None

    def __getitem__(self, idx):
        """
        The proxy speaker is set to none is no voice folder is given.
        @return: data in format (utt_id, path, basename, proxy_speaker)
        """
        utt_id, path, basename = self.data[idx]
        # handle proxy speaker
        if self.spk_to_proxy_spk:
            spk_id = self.spk_from_utt_id(utt_id)
            proxy_speaker = self.spk_to_proxy_spk(spk_id)
        else:
            proxy_speaker = None

        return utt_id, path, basename, proxy_speaker

    def __len__(self):
        return len(self.data)

    def generate_speaker_mapping(self, voice_folder):
        """
        The strategy to assign the speakers is the following:
        First we randomly shuffle the proxy speakers, then we start assigning them to the real speakers.
        If the proxy speakers run out, we re-shuffle them and start over.
        This makes sure that the proxy speaker distribution is as even as possible.

        @param voice_folder: where to take the proxy speakers from.
        @return: A dict of the format {spk_id: proxy_speaker}
        """
        # get the available proxy speakers
        proxy_speakers = os.listdir(voice_folder)  # this assumes the voice folder is organized as required by bark

        # row[0] is always utt_id
        true_speakers = set(self.spk_from_utt_id(row[0]) for row in self.data)
        num_proxy_speakers = len(proxy_speakers)

        # each true speaker picks a random proxy speaker
        speaker_mapping = {}
        random.shuffle(proxy_speakers)
        i = 0
        for speaker in true_speakers:
            speaker_mapping[speaker] = proxy_speakers[i]
            i += 1
            if i == num_proxy_speakers:
                random.shuffle(proxy_speakers)
                i = 0

        return speaker_mapping

    def save_speaker_mapping(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.speaker_mapping, outfile)
