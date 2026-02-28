import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class IndianBirdsDataset(datasets.ImageFolder):
    def __init__(self, root, tokenizer_fn, template="A photo of a", transform=None):
        super().__init__(root, transform)

        self.template = template
        self.tokenizer_fn = tokenizer_fn

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, self.tokenizer_fn(self.template+" "+self.classes[label]).squeeze(0)
