import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DrinksDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.root_path = ('/'.join(csv_path.split('/')[:-1]) + 
                          '/'*(len(csv_path.split('/')[:-1]) > 0))
        self.data_frame = pd.read_csv(csv_path)
        self.image_ids = self.data_frame['frame'].unique()

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_path = self.root_path + self.image_ids[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torchvision.transforms.ToTensor()(image)

        bboxes = self.data_frame[self.data_frame['frame'] == self.image_ids[idx]]
        boxes = bboxes[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = bboxes['class_id'].values

        target = {'boxes':boxes,
                  'labels':torch.as_tensor(labels, dtype=torch.int64),
                  'image_id':torch.tensor([idx]),
                  'area':(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                  'iscrowd':torch.zeros((len(labels),), dtype=torch.int64)}
        
        return image, target