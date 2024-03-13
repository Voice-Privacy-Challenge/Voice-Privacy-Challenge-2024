import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from WGAN.dataset import SpeakerEmbeddingsDataset
from WGAN.training.train_wgan import train_gan
from WGAN.training.logger import setup_logger, setup_tensorboard
from WGAN.init_wgan import create_wgan


def get_comet_logging():
    experiment = Experiment(project_name="WGAN for Speaker Embedding Generation")
    return experiment


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to speaker embeddings",
        default=Path("dataset/reference_embeddings/utt-level/speaker_vectors.pt"),
        type=Path,
    )
    parser.add_argument("--gpu_id", help="GPU to use for training", default=0)
    parser.add_argument("--id", default=None)
    parser.add_argument("--config", type=str, default="./WGAN/configs/train_gan.json")
    parser.add_argument("--comet_ml_experiment_name", type=str, default=None)
    parser.add_argument("--use_comet_ml_logging", action="store_true", default=False)
    args = parser.parse_args()
    return args


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def exit_handler(model, models_dir):
    print(
        f"Stop training at epoch {model.current_epoch} after {model.num_steps} steps!"
    )
    print(
        f'D Loss dist: {np.average(model.losses["D"])}; D Loss asv: {np.average(model.losses["D_ASV"])}; '
        f'Wasserstein distance: {model.losses["WD"][-1]}'
    )
    model._save_models(models_dir)


def main(args):
    logger = setup_logger()
    writer, timestampStr = setup_tensorboard(logger)

    if args.use_comet_ml_logging:
        from comet_ml import Experiment
        experiment = get_comet_logging()
    else:
        experiment = None

    if args.comet_ml_experiment_name:
        experiment.set_name(args.comet_ml_experiment_name)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device '{}'".format(device))

    # parse json parameter file file
    with open(Path(args.config), "r") as f:
        gan_parameters = json.load(f)

    if experiment:
        experiment.log_parameters(args)
        experiment.log_parameters(gan_parameters)

    logger.info("GAN parameters")
    logger.info(gan_parameters)

    gan = create_wgan(parameters=gan_parameters, device=device)
    # atexit.register(exit_handler, gan, Path('models'))

    # create dataset & dataloader
    dataset = SpeakerEmbeddingsDataset(
        feature_path=args.data_path,
        device=device,
        normalize_data=gan_parameters["normalize_data"],
    )
    logger.info("Number of dataset samples:    {:06d}".format(len(dataset)))

    train_gan(
        model=gan,
        dataset=dataset,
        parameters=gan_parameters,
        logger=logger,
        device=device,
        writer=writer,
        timestampStr=timestampStr,
        experiment=experiment,
        save_vis_every=gan_parameters["save_every"]
    )


if __name__ == "__main__":
    main(get_args())