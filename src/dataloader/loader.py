from io import BytesIO
from pandas import read_csv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from src.core.configs import fs


class CustomImageDataset(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None):
        self.img_labels = read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f'{self.img_dir}{self.img_labels.iloc[idx, 0]}'
        image = Image.open(BytesIO(fs.open(img_path).read()))
        image = ToTensor()(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
