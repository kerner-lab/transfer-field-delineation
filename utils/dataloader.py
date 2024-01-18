import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ThreeSeasonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images1 = sorted(os.listdir(os.path.join(root_dir, 'images_mar')))
        self.images2 = sorted(os.listdir(os.path.join(root_dir, 'images_jun')))
        self.images3 = sorted(os.listdir(os.path.join(root_dir, 'images_sep')))
        self.masks_filled = sorted(os.listdir(os.path.join(root_dir, 'masks_filled')))
        self.masks_border = sorted(os.listdir(os.path.join(root_dir, 'masks')))
        self.masks_filled.sort()

    def __len__(self):
        return len(self.masks_filled)

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.root_dir, 'images_mar', self.images1[idx])
        img_name2 = os.path.join(self.root_dir, 'images_jun', self.images2[idx])
        img_name3 = os.path.join(self.root_dir, 'images_sep', self.images3[idx])
        masks_filled = os.path.join(self.root_dir, 'masks_filled', self.masks_filled[idx])
        masks_border = os.path.join(self.root_dir, 'masks', self.masks_border[idx])
        image1 = np.array(Image.open(img_name1))
        image2 = np.array(Image.open(img_name2))
        image3 = np.array(Image.open(img_name3))
        filled = np.array(Image.open(masks_filled))
        border = np.array(Image.open(masks_border))

        image = np.concatenate((image1, image2, image3), axis=2)
        
        # Convert to tensor
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        filled = np.array(filled)
        filled = np.expand_dims(filled, axis=0)
        filled[filled > 0] = 1
        filled = filled.astype(np.float32)

        border = np.array(border)
        border = np.expand_dims(border, axis=0)
        border[border > 0] = 1
        border = border.astype(np.float32)

        sample = {'image': image, 'filled': filled, 'border': border}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    dataset = ThreeSeasonDataset(root_dir='data/data/sentinel/')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['filled'].size(), sample_batched['border'].size())
        if i_batch == 3:
            print(sample_batched['filled'])
            break