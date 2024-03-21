import torch
from pathlib import Path
from .SpeakerVisualization import Visualizer
import PIL


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def inverse_normalize_v2(tensor, mean, std):
    return tensor * std + mean


def train_gan(
    model,
    dataset,
    parameters,
    logger,
    device,
    writer,
    timestampStr,
    experiment,
    save_vis_every=2,
):
    # train gan
    train_data_len = int(len(dataset) * 0.8)
    dev_data_len = int(len(dataset) - train_data_len)
    train_data, dev_data = torch.utils.data.random_split(
        dataset, (train_data_len, dev_data_len)
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=parameters["batch_size"], drop_last=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, shuffle=True, batch_size=parameters["batch_size"]
    )

    samples_dir = Path("generated_sample_embeddings")
    samples_dir.mkdir(exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for cycle in range(parameters["cycles"]):
        logger.info(f"Start cycle {cycle}")
        logger.info(f"Model trained for {model.num_steps} iterations")
        if model.num_steps >= parameters["n_max_iterations"]:
            logger.info("Breaky cycle loop - Max iterations reached - Stop training")
            break

        model.G.train()
        model.D.train()

        model.train(train_loader, writer, experiment)

        model.G.eval()
        model.D.eval()

        if (cycle + 1) % save_vis_every == 0:
            logger.info("Generating samples from WGAN")
            samples_generated = inverse_normalize_v2(
                model.sample_generator(num_samples=10, nograd=True).cpu(),
                train_loader.dataset.dataset.mean.unsqueeze(0).cpu(),
                train_loader.dataset.dataset.std.unsqueeze(0).cpu(),
            )
            logger.info("Saving samples")
            Path('./test_samples').mkdir(exist_ok=True)
            torch.save(
                samples_generated,
                f"./test_samples/{timestampStr}_generated_samples_num_steps_{model.num_steps}.pt",
            )
            logger.info("Saving model")
            model.save_model_checkpoint(
                models_dir, parameters, timestampStr, dataset.mean, dataset.std
            )
            logger.info("Model saved")

    logger.info("Save model")
    model.save_model_checkpoint(
        models_dir, parameters, timestampStr, dataset.mean, dataset.std
    )
    logger.info("Model saved")