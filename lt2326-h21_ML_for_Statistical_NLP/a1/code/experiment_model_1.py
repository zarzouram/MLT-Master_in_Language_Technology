# %% [markdown]

# %%
from pathlib import Path
from tqdm.notebook import tqdm

from sklearn.metrics import classification_report

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.simple_model_1 import Model_1
from dataset.loader import HDF5Dataset

# %%
# source: https://github.com/NVIDIA/framework-determinism/blob/master/pytorch.md

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
CUDA_LAUNCH_BLOCKING = 1

# %% [markdown]
# ## Initialize some variables

# %%
dataset_dir = "/scratch/guszarzmo/lt2326_labs/lab1/data"
model_dir = "/scratch/guszarzmo/lt2326_labs/lab1/checkpoints"

# %%
# Variable changed by user
checkpoint = "m1_1210_20h5501.pt"
use_cpu = False  # set to True if you want to use cpu
prefered_device = None  # gpu id

# %% [markdown]
#  ## Selecte device

# %%
# Select cuda device based on the free memory
from utils.gpu_cuda_helper import select_device

if use_cpu:
    device = torch.device("cpu")
    print(f"\ndevice selected: {device}")

elif prefered_device:
    device = select_device(prefered_device)
    print(f"\ndevice selected: {device}")

else:
    device = select_device(-1)

# %% [markdown]
# ## Initialize dataset and dataloader

# %%

pin_memory = device.type == "cuda"
kwargs = dict(shuffle=False, num_workers=4, pin_memory=pin_memory)

test_ds_path = str(Path(dataset_dir) / "train.h5")
test_ds = HDF5Dataset(test_ds_path)
test_iter = DataLoader(test_ds, batch_size=1, **kwargs)

# %% [markdown]
# ## Model loading

# %%
model = Model_1().to(device)

load_path = Path(model_dir) / "model1" / checkpoint
model_data = torch.load(load_path, map_location=torch.device("cpu"))
model_state_dict = model_data["model_state_dict"]

model.load_state_dict(model_state_dict)

# %% [markdown]
# ## Testing

# %%
transform = transforms.Normalize(*test_ds.mean_stddev)  # normalization
criterion = nn.CrossEntropyLoss().to(device)
class_names = ["n-char", "is_chinese", "ignore", "char"]

model.eval()
test_pb = tqdm(total=len(test_ds), unit="image")  # progress bar
losses = []

record_steps = []
f1_scores = []
f1_score_ischinese = []
f1_score_char = []
precisions = []
recalls = []
images = []
predections = []
for step, (images, labels) in enumerate(test_iter):
    test_pb.set_description(f"Step: {step:<3d}")
    images = transform(images.to(device))  # type: Tensor
    labels = labels.to(device)  # type: Tensor
    with torch.no_grad():
        out = model(images)
        loss = criterion(out, labels)

    losses.append(loss.item())
    test_pb.update(1)

    probs = torch.log_softmax(out, dim=1)
    _, preds = torch.max(probs, dim=1)

    target_names = [
        class_names[i] for i in range(4)
        if (labels == i).any() or (preds == i).any()
    ]
    scores = classification_report(labels.detach().cpu().view(-1),
                                   preds.detach().cpu().view(-1),
                                   target_names=target_names,
                                   output_dict=True,
                                   zero_division=0)

    f1_scores.append(scores["macro avg"]["f1-score"])
    precisions.append(scores["macro avg"]["precision"])
    recalls.append(scores["macro avg"]["recall"])

    f1_score_ischinese.append(scores["is_chinese"]["f1-score"])
    if "char" in scores:
        f1_score_char.append(scores["char"]["f1-score"])

# %% [markdown]
# ## Testing stats

# %%
print(
    f"f1 score:  {np.array(f1_scores).mean(): .3f} \u00B1 {np.array(f1_scores).std(): .3f}"  # noqa: E501
)

print(
    f"Precision: {np.array(precisions).mean(): .3f} \u00B1 {np.array(precisions).std(): .3f}"  # noqa: E501
)

print(
    f"Recall:    {np.array(recalls).mean(): .3f} \u00B1 {np.array(recalls).std(): .3f}"  # noqa: E501
)

# %%
print(
    f"is_chinese class - f1 score:  {np.array(f1_score_ischinese).mean(): .3f} \u00B1 {np.array(f1_score_ischinese).std(): .3f} "  # noqa: E501
)

print(
    f"char       class - f1 score:  {np.array(f1_score_char).mean(): .3f} \u00B1 {np.array(f1_score_char).std(): .3f} "  # noqa: E501
)

# %%
