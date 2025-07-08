from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import io


class TaskDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.imgs_paths = []
        self.ques = []
        self.ans = []

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_paths[idx])
        img = np.array(img)
        ques = self.ques[idx]
        ans = self.ans[idx]

        if self.transform:
            img = self.transform(img)
        return img, ques, ans

    def load_data(self):
        pass


class VQADataset(TaskDataset):
    def __init__(self, data_path, datatype, transform=None):
        super().__init__(data_path, transform)
        self.load_data(datatype)

    def load_data(self, datatype):
        df = pd.read_parquet(os.path.join(self.data_path, f"{datatype}.parquet"))
        self.imgs_paths = df["image_id"].tolist()
        self.imgs_paths = [
            os.path.join(self.data_path, "images", f"COCO_val2014_{path:012d}.jpg")
            for path in self.imgs_paths
        ]
        self.ques = df["question"].tolist()
        self.ans = df["answers"].tolist()


class RSVQADataset(TaskDataset):
    def __init__(self, data_path, datatype, transform=None):
        super().__init__(data_path, transform)
        self.load_data(datatype)

    def load_data(self, datatype):
        df = pd.read_parquet(os.path.join(self.data_path, f"{datatype}.parquet"))
        self.imgs_paths = df["id"].tolist()
        self.imgs_paths = [
            os.path.join(self.data_path, "images", f"{path}.png")
            for path in self.imgs_paths
        ]
        self.ques = df["question"].tolist()
        self.ans = df["answer"].tolist()


class PMCVQADataset(TaskDataset):
    def __init__(self, data_path, datatype, transform=None):
        super().__init__(data_path, transform)
        self.load_data(datatype)

    def get_correct_ans(self, df):
        correct_ans = []
        for i, ans in enumerate(self.ans):
            col_name = f"Choice {ans}"
            correct_ans.append(df.iloc[i, df.columns.get_loc(col_name)])
        return correct_ans

    def load_data(self, datatype):
        df = pd.read_parquet(os.path.join(self.data_path, f"{datatype}.parquet"))
        self.imgs_paths = df["Figure_path"].tolist()
        self.imgs_paths = [
            os.path.join(self.data_path, "images", f"{path}")
            for path in self.imgs_paths
        ]
        self.ques = df["Question"].tolist()
        self.ans = df["Answer"].tolist()
        self.ans = self.get_correct_ans(df)


class LingoQADataset(TaskDataset):
    def __init__(self, data_path, datatype, transform=None):
        super().__init__(data_path, transform)
        self.load_data(datatype)

    def load_data(self, datatype):
        df = pd.read_parquet(os.path.join(self.data_path, f"{datatype}.parquet"))
        self.imgs_paths = df["images"].tolist()
        self.ques = df["question"].tolist()
        self.ans = df["answer"].tolist()

    def __getitem__(self, idx):
        img_list = []
        for img_path in self.imgs_paths[idx]:
            img = Image.open(os.path.join(self.data_path, img_path))
            img = np.array(img)
            if self.transform:
                img = self.transform(img)
            img_list.append(img)
        return img_list, self.ques[idx], self.ans[idx]


class docVQADataset(TaskDataset):
    def __init__(self, data_path, datatype, transform=None):
        super().__init__(data_path, transform)
        self.load_data(datatype)

    def load_data(self, datatype):
        df = pd.read_parquet(os.path.join(self.data_path, f"{datatype}.parquet"))
        self.imgs_paths = df["id"].tolist()
        self.imgs_paths = [
            os.path.join(self.data_path, "images", f"{path:012d}.png")
            for path in self.imgs_paths
        ]
        self.ques = df["question"].tolist()
        self.ans = df["answer"].tolist()

    def __getitem__(self, idx):
        Image.open(io.BytesIO(df.iloc[0]["image"]["bytes"]))
        img = Image.open(self.imgs_paths[idx])
        img = np.array(img)
        ques = self.ques[idx]
        ans = self.ans[idx]

        if self.transform:
            img = self.transform(img)
        return img, ques, ans
