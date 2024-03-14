import argparse
import os
import random
import torch
from torch.utils.data import ConcatDataset

from IMSToucan.TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from IMSToucan.TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from IMSToucan.Utility.corpus_preparation import prepare_fastspeech_corpus


def limit_to_n(path_to_transcript_dict, n=30000):
    limited_dict = dict()
    if len(path_to_transcript_dict.keys()) > n:
        for key in random.sample(list(path_to_transcript_dict.keys()), n):
            limited_dict[key] = path_to_transcript_dict[key]
        return limited_dict
    else:
        return path_to_transcript_dict


def build_path_to_transcript_dict_libritts_clean(train_data_path):
    # this is fitted to the LibriTTS data structure, for a different data set, you need to adjust this function
    path_to_transcript = dict()
    for speaker in os.listdir(train_data_path):
        for chapter in os.listdir(os.path.join(train_data_path, speaker)):
            for file in os.listdir(os.path.join(train_data_path, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(train_data_path, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(train_data_path, speaker, chapter, wav_file)] = transcript
    return limit_to_n(path_to_transcript)


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, train_data_path):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Controllable")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    libri_transcript_dict = build_path_to_transcript_dict_libritts_clean(train_data_path)
    datasets.append(prepare_fastspeech_corpus(transcript_dict=libri_transcript_dict,
                                              corpus_dir=os.path.join("Corpora", "libri_clean"),
                                              lang="en"))

    train_set = ConcatDataset(datasets)

    model = FastSpeech2(lang_embs=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=32,
               lang="en",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=4000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               phase_1_steps=50000,
               phase_2_steps=50000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMS Speech Synthesis Toolkit - Call to Train')

    parser.add_argument('--train_data_path',
                        type=str,
                        help="Path to train data.",
                        default="corpora/LibriTTS/clean")

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--resume_checkpoint',
                        type=str,
                        help="Path to checkpoint to resume from.",
                        default=None)

    parser.add_argument('--resume',
                        action="store_true",
                        help="Automatically load the highest checkpoint and continue from there.",
                        default=False)

    parser.add_argument('--finetune',
                        action="store_true",
                        help="Whether to fine-tune from the specified checkpoint.",
                        default=False)

    parser.add_argument('--model_save_dir',
                        type=str,
                        help="Directory where the checkpoints should be saved to.",
                        default=None)

    args = parser.parse_args()

    run(train_data_path=args.train_data_path,
        gpu_id=args.gpu_id,
        resume_checkpoint=args.resume_checkpoint,
        resume=args.resume,
        finetune=args.finetune,
        model_dir=args.model_save_dir)
