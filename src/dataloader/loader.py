from io import BytesIO
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
from typing import Tuple
from PIL import Image
from core.configs import fs, model_settings
from utils import splitted_dataframes

IMG_SIZE = 128


class CustomImageDataset(Dataset):
    def __init__(
            self,
            img_labels: DataFrame,
            img_dir: str,
            transform=Compose([ToTensor()]),
            target_transform=None
    ) -> None:
        super().__init__()
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_path = f'{self.img_dir}{self.img_labels.iloc[idx, 0]}'
        image = Image.open(BytesIO(fs.open(img_path).read()))
        image = image.resize((IMG_SIZE, IMG_SIZE))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MaskDataloaders:
    def __init__(self, img_dir: str, split_fractions: list):
        (
            self.trainDataset,
            self.validationDataset
        ) = tuple(
            map(
                lambda x: CustomImageDataset(x, img_dir),
                splitted_dataframes(split_fractions)
            )
        )

    def getDataloaders(self):
        return (
            DataLoader(
                self.trainDataset,
                batch_size=model_settings.batch_size,
                shuffle=model_settings.shuffle
            ),
            DataLoader(
                self.validationDataset,
                batch_size=model_settings.batch_size,
                shuffle=model_settings.shuffle
            )

        )
