from .base_anon import BaseAnonymizer
import torch
from typing import Union


class Passthrough(BaseAnonymizer):
    """
    A 'Passthrough' 'anonymizer' that does not anonymize the speaker embeddings.
    """

    def __init__(
        self,
        vec_type: str = "",
        device: Union[str, torch.device, int] = "cuda:0",
        suffix: str = "_res",
        **kwargs
    ):
        super().__init__(vec_type, device, suffix, **kwargs)

    def anonymize_embeddings(
        self, speaker_embeddings: torch.Tensor, emb_level: str = "spk"
    ) -> torch.Tensor:
        """
        Returns the speaker embeddings unchanged.
        """
        # no need to refer to emb_level,
        # as extractor also yields spk-level or utt-level.
        return speaker_embeddings

    def to(self, device):
        """
        Move the anonymizer to the given device. For the passthrough anonymizer,
        this is a no-op, apart from setting the property.
        """
        super().to(device)
