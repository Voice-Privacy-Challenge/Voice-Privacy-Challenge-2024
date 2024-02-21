import torch


class ImsProsodyAnonymization:

    def __init__(self, random_offset_lower, random_offset_higher):
        self.random_offset_lower = random_offset_lower
        self.random_offset_higher = random_offset_higher

    def anonymize_values(self, duration, energy, pitch, *kwargs):
        if self.random_offset_lower is not None and self.random_offset_higher is not None:
            scales = torch.randint(low=self.random_offset_lower, high=self.random_offset_higher,
                                   size=energy.size()).float() / 100
            energy = energy * scales

        if self.random_offset_lower is not None and self.random_offset_higher is not None:
            scales = torch.randint(low=self.random_offset_lower, high=self.random_offset_higher,
                                   size=pitch.size()).float() / 100
            pitch = pitch * scales

        return_dict = {
            'duration': duration,
            'energy': energy,
            'pitch': pitch
        }

        return_dict.update(kwargs)

        return return_dict
