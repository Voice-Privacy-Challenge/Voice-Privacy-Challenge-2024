# Disclaimer: To satisfy challenge requirements, this script assumes to train the vocoder only on LibriTTS
# However, it is generally better to train it on much more data
import argparse
import os
import random
import torch
from torch.utils.data import ConcatDataset

from IMSToucan.TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANGenerator
from IMSToucan.TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANMultiScaleMultiPeriodDiscriminator
from IMSToucan.TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGANDataset import HiFiGANDataset
from IMSToucan.TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.hifigan_train_loop import train_loop


def get_file_list_libritts(train_data_path):
    file_list = list()
    for speaker in os.listdir(train_data_path):
        for chapter in os.listdir(os.path.join(train_data_path, speaker)):
            for file in os.listdir(os.path.join(train_data_path, speaker, chapter)):
                if file.endswith(".wav"):
                    file_list.append(os.path.join(train_data_path, speaker, chapter, file))
    return file_list


def run(gpu_id, resume_checkpoint, finetune, resume, model_dir, train_data_path):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")
    if model_dir is not None:
        model_save_dir = model_dir
    else:
        model_save_dir = "Models/HiFiGAN_combined"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # sampling multiple times from the dataset, because it's too big to fit all at once
    for run_id in range(800):

        file_list = random.sample(get_file_list_libritts(train_data_path), 5000)
        train_set = HiFiGANDataset(list_of_paths=file_list, cache_dir=f"Corpora/0", use_random_corruption=True)

        generator = HiFiGANGenerator()
        generator.reset_parameters()
        discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator()

        print("Training model")
        if run_id == 0:
            train_loop(batch_size=16,
                       epochs=20,
                       generator=generator,
                       discriminator=discriminator,
                       train_dataset=train_set,
                       device=device,
                       epochs_per_save=2,
                       model_save_dir=model_save_dir,
                       path_to_checkpoint=resume_checkpoint,
                       resume=resume,
                       use_signal_processing_losses=False)
        else:
            train_loop(batch_size=16,
                       epochs=20,
                       generator=generator,
                       discriminator=discriminator,
                       train_dataset=train_set,
                       device=device,
                       epochs_per_save=2,
                       model_save_dir=model_save_dir,
                       path_to_checkpoint=None,
                       resume=True,
                       use_signal_processing_losses=False)

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