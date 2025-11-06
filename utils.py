from braindecode.datasets import BaseConcatDataset
import torch

def split_by_subjects(dataset, subject_list):
    indices = []
    for ds in dataset.datasets:
        subject = ds.windows_ds.description['subject']
        if subject in subject_list:
            indices.append(ds)

    return BaseConcatDataset(indices)

def collate_fn_challenge2(batch):
    X, meta, y, _, _ = zip(*batch)

    return (
        torch.stack(X),
        torch.stack(meta),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )

