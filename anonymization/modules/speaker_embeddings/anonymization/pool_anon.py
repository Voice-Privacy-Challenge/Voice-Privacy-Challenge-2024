from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm
from typing import Union

from os import PathLike
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import minmax_scale, StandardScaler

from .base_anon import BaseAnonymizer
from .utils.plda_model import PLDAModel
from ..speaker_extraction import SpeakerExtraction
from ..speaker_embeddings import SpeakerEmbeddings
from utils import transform_path, setup_logger

logger = setup_logger(__name__)

REVERSED_GENDERS = {
    "m": "f",
    "f": "m"
}


class PoolAnonymizer(BaseAnonymizer):
    """
    An implementation of the 'Pool' anonymization method, that is based on the
    primary baseline of the Voice Privacy Challenge 2020.

    For every source x-vector, an anonymized x-vector is computed by finding
    the N farthest x-vectors in an external pool (LibriTTS train-other-500)
    according to the PLDA distance, and by averaging N∗ randomly selected
    vectors among them. In the baseline, we use:
        N = 200,
        N∗ = 100.
    """
    def __init__(
        self,
        vec_type: str = "xvector",
        device: Union[str, torch.device, int, None] = None,
        model_name: str = None,
        pool_data_dir: Union[str, PathLike] = "data/libritts_train_other_500",
        pool_vec_path: Union[str, PathLike] = "original_speaker_embeddings/pool_embeddings",
        N: int = 200,
        N_star: int = 100,
        distance: str = "plda",
        cross_gender: bool = False,
        proximity: bool = "farthest",
        scaling: str = None,
        stats_per_dim_path: Union[str, PathLike] = None,
        distance_model_path: Union[str, PathLike] = "distances/plda/libritts_train_other_500_xvector",
        emb_model_path: Union[str, PathLike] = None,
        save_intermediate: bool = False,
        suffix: str = "_anon",
        **kwargs,
    ):
        """
        Args:
            vec_type (str): Type of the speaker embeddings, currently supported
                are 'xvector', 'ecapa', 'style-embed'.

            device (Union[str, torch.device, int, None]): Device to use for
                the procedure, e.g. 'cpu', 'cuda', 'cuda:0', etc.

            model_name (str): Name of the model, used for distances that
                require a model (e.g., PLDA).

            pool_data_dir (Union[str, PathLike]): Path to the audio data
            which will be used for x-vector pool extraction.

            pool_vec_path (Union[str, PathLike]): Path to the stored
                speaker embeddings of the pool.

            N (int): Number of most 'fitting' vectors to consider.

            N_star (int): Number of vectors to randomly select from the N most
                'fitting' vectors, to compute the average.

            distance (str): Distance measure, either 'plda' or 'cosine'.

            cross_gender (bool): Whether to switch genders of the speakers
                during anonymization.

            proximity (str): Proximity measure, determining which vectors in
                the pool are the 'fittest', can be either 'farthest',
                'nearest' or 'center'.

            scaling (str): Scaling method to use, can be either 'minmax' or
                'std'.

            stats_per_dim_path (Union[str, PathLike]): Path to the file
                containing the statistics for each dimension in the given
                embedding type.

            distance_model_path (Union[str, PathLike]): Path to the stored
                distance model (required for PLDA).

            emb_model_path (Union[str, PathLike]): Path to the directory
                containing the speaker embedding model.

            save_intermediate (bool): Whether to save intermediate results.

            suffix (str): Suffix to append to the output folder names.

        """
        # filter out kwargs that are not used in this class
        kwargs = locals()
        kwargs.pop("self")
        kwargs.pop("__class__")
        super().__init__(**kwargs)

        self.model_name = model_name if model_name else f"pool_{vec_type}"

        self.N = N
        self.N_star = N_star
        self.proximity = proximity
        self.cross_gender = cross_gender
        self.save_intermediate = save_intermediate

        # external speaker pool
        self.pool_embeddings = self._load_pool_embeddings(
            pool_data_dir=Path(pool_data_dir).expanduser(),
            pool_vec_path=Path(pool_vec_path).expanduser(),
            emb_model_path=Path(emb_model_path).expanduser(),
        )
        self.pool_genders = {
            gender: [
                i
                for i, spk_gender in enumerate(self.pool_embeddings.genders)
                if spk_gender == gender
            ]
            for gender in set(self.pool_embeddings.genders)
        }

        # distance model; PLDA model if distance == plda; None if distance == cosine
        self.distance = distance  # distance measure, either 'plda' or 'cosine'
        if self.distance == "plda":
            self.distance_model = PLDAModel(
                train_embeddings=self.pool_embeddings,
                results_path=Path(distance_model_path),
                save_plda=self.save_intermediate,
            )
        else:
            self.distance_model = None

        # scaling to ensure correct value ranges per dimension
        self.scaling = scaling
        self.stats_per_dim_path = stats_per_dim_path or Path()

    def _load_pool_embeddings(self, pool_data_dir, pool_vec_path, emb_model_path):
        logger.debug(pool_data_dir)
        if pool_vec_path.exists():
            pool_embeddings = SpeakerEmbeddings(
                vec_type=self.vec_type, emb_level="spk", device=self.device
            )
            pool_embeddings.load_vectors(pool_vec_path)
        else:
            extraction_settings = {"vec_type": self.vec_type, "emb_level": "spk", "emb_model_path": emb_model_path}
            emb_extractor = SpeakerExtraction(
                results_dir=pool_vec_path,
                devices=[self.device],
                settings=extraction_settings,
                save_intermediate=self.save_intermediate,
            )
            pool_embeddings = emb_extractor.extract_speakers(
                dataset_path=pool_data_dir, dataset_name=""
            )
        return pool_embeddings

    def anonymize_embeddings(self, speaker_embeddings: torch.Tensor, emb_level: str = "spk"):
        distance_matrix = self._compute_distances(
            vectors_a=self.pool_embeddings.vectors, vectors_b=speaker_embeddings.vectors
        )

        logger.info(f"Anonymize embeddings of {len(speaker_embeddings)} speakers...")
        identifiers = []
        speakers = []
        anon_vectors = []
        genders = []

        for i in tqdm(range(len(speaker_embeddings))):
            identifier, _ = speaker_embeddings[i]
            speaker = speaker_embeddings.original_speakers[i]
            gender = speaker_embeddings.genders[i]
            distances_to_speaker = distance_matrix[:, i]
            candidates = self._get_pool_candidates(distances_to_speaker, gender)
            selected_anon_pool = np.random.choice(
                candidates, self.N_star, replace=False
            )
            anon_vec = torch.mean(
                self.pool_embeddings.speaker_vectors[selected_anon_pool], dim=0
            )
            identifiers.append(identifier)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)
            genders.append(
                gender if not self.cross_gender else REVERSED_GENDERS[gender]
            )

        anon_embeddings = SpeakerEmbeddings(
            vec_type=self.vec_type, device=self.device, emb_level=emb_level
        )
        anon_embeddings.set_vectors(
            identifiers=identifiers,
            vectors=torch.stack(anon_vectors, dim=0),
            speakers=speakers,
            genders=genders,
        )

        return anon_embeddings

    def _compute_distances(self, vectors_a, vectors_b):
        if self.distance == "plda":
            return 1 - self.distance_model.compute_distance(
                enrollment_vectors=vectors_a, trial_vectors=vectors_b
            )
        elif self.distance == "cosine":
            return cosine_distances(X=vectors_a.cpu(), Y=vectors_b.cpu())
        else:
            return []

    def _get_pool_candidates(self, distances, gender):
        if self.cross_gender is True:
            distances = distances[self.pool_genders[REVERSED_GENDERS[gender]]]
        else:
            distances = distances[self.pool_genders[gender]]

        if self.proximity == "farthest":
            return np.argpartition(distances, -self.N)[-self.N :]
        elif self.proximity == "nearest":
            return np.argpartition(distances, self.N)[: self.N]
        elif self.proximity == "center":
            sorted_distances = np.sort(distances)
            return sorted_distances[
                len(sorted_distances) // 2 : (len(sorted_distances) // 2) + self.N
            ]

    def _load_scaling_ranges(self, stats_per_dim_path):
        if stats_per_dim_path and Path(stats_per_dim_path).exists():
            with open(stats_per_dim_path) as f:
                dim_ranges = json.load(f)
                return [
                    (v["min"], v["max"])
                    for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))
                ]
        else:
            raise FileNotFoundError(
                f"You need to specify a path to an existing file containing the statistics for "
                f"each dimension in the given embedding type, "
                f"stats_per_dim_path={stats_per_dim_path} is not valid!"
            )

    def _scale_embeddings(self, embeddings):
        vectors = embeddings.vectors.cpu().numpy()

        if self.scaling == "minmax":
            scaling_ranges = self._load_scaling_ranges(self.stats_per_dim_path)
            scaled_dims = []
            for i in range(len(scaling_ranges)):
                scaled_dims.append(
                    minmax_scale(vectors[:, i], scaling_ranges[i], axis=0)
                )

            scaled_vectors = torch.tensor(np.array(scaled_dims)).T.to(self.device)
            embeddings.vectors = scaled_vectors

        elif self.scaling == "std":
            std_scaler = StandardScaler()
            std_scaler.fit(self.pool_embeddings.vectors.cpu().numpy())
            scaled_vectors = torch.tensor(std_scaler.transform(vectors))
            embeddings.vectors = scaled_vectors

        return embeddings
