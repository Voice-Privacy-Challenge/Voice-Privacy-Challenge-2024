from pathlib import Path
import torch
import ruamel.yaml as yaml
from ruamel.yaml.representer import RoundTripRepresenter, SafeRepresenter
from typing import Union


class BaseAnonymizer:
    """
    Base class for speaker embedding anonymizers, defining the API,
    that consists of the following methods:
    - anonymize_embeddings
    - to
    """
    def __init__(
        self,
        vec_type: str,
        device: Union[str, torch.device, int, None],
        suffix: str,
        **kwargs,
    ):
        assert suffix[0] == "_", "Suffix must be a string and start with an underscore."

        # Base class for speaker embedding anonymization.
        self.vec_type = vec_type
        self.suffix = suffix

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # ensure dumpability
        self.kwargs = kwargs
        self.kwargs["vec_type"] = self.vec_type
        self.kwargs["device"] = str(self.device)
        self.kwargs["suffix"] = self.suffix

    def __repr__(self):
        if hasattr(self, "kwargs"):
            return f"{self.__class__.__name__}({self.kwargs})"
        else:
            return f"{self.__class__.__name__}()"

    def to_yaml(self, representer: yaml.Representer):
        # first get data into dict format
        data = {f"!new:{type(self).__qualname__}": self.kwargs}
        return_str = representer.represent_dict(data)
        return return_str

    def anonymize_embeddings(self, speaker_embeddings: torch.Tensor, emb_level: str = "spk") -> torch.Tensor:
        # Template method for anonymizing a dataset. Not implemented.
        raise NotImplementedError("anonymize_data")

    def to(self, device):
        self.device = device


# necessary to make BaseAnonymizer and subclasses dumpable
RoundTripRepresenter.add_multi_representer(
    BaseAnonymizer, lambda representer, data: data.to_yaml(representer)
)
