from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import typing

class MetaEncoder:
    def __init__(self):
        self.task_enc = LabelEncoder()
        self.sex_enc = LabelEncoder()
        self.age_scaler = StandardScaler()

    def fit(self, metas: typing.List[dict]):
        tasks = [m["task"] for m in metas]
        sexes = [m["sex"] for m in metas]
        ages = [[m["age"]] for m in metas]
        self.task_enc.fit(tasks)
        self.sex_enc.fit(sexes)
        self.age_scaler.fit(ages)
        self.dim = len(self.task_enc.classes_) + len(self.sex_enc.classes_) + 1
        return self

    def transform(self, meta: dict) -> torch.Tensor:
        t = self.task_enc.transform([meta["task"]])[0]
        s = self.sex_enc.transform([meta["sex"]])[0]
        a = self.age_scaler.transform([[meta["age"]]])[0, 0]
        vec = torch.zeros(self.dim, dtype=torch.float32)
        vec[t] = 1.0
        vec[len(self.task_enc.classes_) + s] = 1.0
        vec[-1] = a
        return vec