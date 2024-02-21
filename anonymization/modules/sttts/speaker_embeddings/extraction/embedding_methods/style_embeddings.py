import warnings
import torch
from ....tts.IMSToucan.TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from ....tts.IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor


class StyleEmbeddings:

    def __init__(self, model_path, device):
        self.device = device

        self.extractor = StyleEmbedding()
        check_dict = torch.load(model_path, map_location='cpu')
        self.extractor.load_state_dict(check_dict['style_emb_func'])
        self.extractor.to(self.device)

        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True,
                                                    device=self.device)

    def extract_vector(self, audio, sr):
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True,
                                                        device=self.device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = self.audio_preprocessor.cut_silence_from_audio(audio.to(self.device)).cpu()
            spec = self.audio_preprocessor.logmelfilterbank(audio, 16000).transpose(0, 1)
            spec_len = torch.LongTensor([len(spec)])
            vector = self.extractor(spec.unsqueeze(0).to(self.device), spec_len.unsqueeze(0).to(self.device))
        return vector.squeeze().detach()
