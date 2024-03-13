from typing import Union

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from encodec.utils import convert_audio

from TTS.tts.layers.bark.inference_funcs import semantic_tokens_from_audio, load_voice
import torch
import torchaudio
from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer

import os


class Anonymizer(torch.nn.Module):
    def __init__(self, checkpoint_dir: Union[str, None] = None, voice_dirs: Union[list[str], None] = None):
        super().__init__()

        if checkpoint_dir is None:
            checkpoint_dir = 'exp/nac_models'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 1. initialize Bark
        config = BarkConfig()  # don't change the custom config for the love of god
        self.model = Bark.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)
        # self.model.to('cuda')

        # 2. initialize the awesome, bark-distilled, unlikely-yet-functioning audio tokenizer
        hubert_manager = HubertManager()
        hubert_manager.make_sure_tokenizer_installed(model_path=self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"])
        self.hubert_model = CustomHubert(
            checkpoint_path=self.model.config.LOCAL_MODEL_PATHS["hubert"])  # .to(self.model.device)
        self.tokenizer = HubertTokenizer.load_from_checkpoint(
            self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"], map_location=self.model.device
        )

        self.voice_dirs = voice_dirs
        self.sample_rate = self.model.config.sample_rate

    def forward(
            self,
            audio: Union[torch.Tensor, str],
            target_voice_id: str = 'random',
            coarse_temperature: float = 0.7
    ):
        # You can give the audio as path to a wav. In this case, resampling and reshaping is done
        # If you directly give a tensor: must be 1 channel, 24k sr, and shape (1, L)
        # batched inference is currently not supported, sorry
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
            audio = convert_audio(audio, sr, self.model.config.sample_rate, self.model.encodec.channels)
            audio = audio.to(
                self.model.device)  # there used to be an unsqueeze here but then they squeeze it back so it's useless

        # 1. Extraction of semantic tokens
        semantic_vectors = self.hubert_model.forward(audio, input_sample_hz=self.model.config.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors)
        semantic_tokens = semantic_tokens.cpu().numpy() # they must be shifted to cpu
        # this probably slows things down, but the following api function from bark specifically requires numpy
        # but i mean, what the fuck do i know

        # 2. Load voice as a history prompt as a tuple (semantic_prompt, coarse_prompt, fine_prompt)
        if not self.voice_dirs:
            assert target_voice_id == 'random', """If no voice dirs are given, the target voice must be 'random'.
            Note that, regardless of this, 'random' always means 'use an empty semantic and coarse prompts'.
            So even if target_voice_id == 'random', the voice_dirs will be ignored (it does NOT mean it will pick a
            random voice from there).
            ...this should probably go into some documentation. Why am I writing it here?"""
        history_prompt = load_voice(self.model, target_voice_id, self.voice_dirs)

        # 3. Regression of acoustic tokens with bark api
        # 'temp' here is only the coarse temperature. The fine temperature is internally fixed to 0.5
        # (i fiddled with it a bit and it does seem a bit of a sweet spot, any higher and the audio gets a bit dirty)
        # the other two returned values are coarse and fine tokens, we don't need them for now
        audio_arr, _, _ = self.model.semantic_to_waveform(
            semantic_tokens, history_prompt=history_prompt, temp=coarse_temperature
        )
        return audio_arr
