import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path('../../../../../..').resolve().absolute()))
from anonymization.modules.sttts.speaker_embeddings.speaker_extraction import SpeakerExtraction


def extract_train_embeddings(data_path, save_dir, gpu_id, emb_model_path, vec_type, emb_level):
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() and gpu_id != 'cpu' else torch.device('cpu')
    settings = {
        'emb_model_path': Path(emb_model_path),
        'vec_type': vec_type,
        'emb_level': emb_level
    }
    extractor = SpeakerExtraction(devices=[device], settings=settings, results_dir=Path(save_dir),
                                  save_intermediate=True, force_compute=True)
    extractor.extract_speakers(dataset_path=Path(data_path), dataset_name='reference_embeddings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Speaker Embeddings for GAN Training')

    parser.add_argument('--data_path', type=str, help="Path to train data in kaldi format.",
                        default='data/train-clean-100')
    parser.add_argument('--save_dir', type=str,
                        help="Directory where the extracted speaker embeddings should be saved to.",
                        default='dataset')
    parser.add_argument('--gpu_id', type=str, help="Which GPU to run on. Will run on CPU if not specified.",
                        default="cpu")
    parser.add_argument('--emb_model_path', type=str, help="Location of embedding extraction model.",
                        default="../../../../../../exp/sttts_models/tts/Embedding/embedding_function.pt")
    parser.add_argument('--vec_type', type=str, help="Which vector type to use for extraction, e.g. style-embed, "
                                                     "ecapa, ecapa+xvector, ...",
                        default="style-embed")
    parser.add_argument('--emb_level', type=str, help="Whether to extract the embeddings on utterance-level or "
                                                      "speaker-level",
                        choices=['utt', 'spk'],
                        default="utt")

    args = parser.parse_args()
    extract_train_embeddings(args.data_path, args.save_dir, args.gpu_id, args.emb_model_path, args.vec_type,
                             args.emb_level)