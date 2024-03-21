import torch
import numpy as np

class NoiseEmbeddingsDataset(torch.utils.data.Dataset):

    def __init__(self, latent_dim, num_embeddings, device, label=None):
        super(NoiseEmbeddingsDataset, self).__init__()

        self.device = device

        self.x = torch.randn(num_embeddings, latent_dim, device=device)

        #if label is not None:
        self.y = torch.full(size=(num_embeddings,), fill_value=label).to(self.device)
        #else:
        #    self.y = torch.arange(num_embeddings).to(self.device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]