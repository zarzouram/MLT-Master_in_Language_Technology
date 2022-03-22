from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torchvision.transforms import transforms

from sklearn.metrics import f1_score

from models.simple_model_1 import Model_1
from models.simple_model_2 import Model_2
from dataset.loader import HDF5Dataset
from utils.gpu_cuda_helper import select_device
from utils.visualization import Visualizations
from utils.scheduler import IncreaseLROnPlateau

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
CUDA_LAUNCH_BLOCKING = 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="LT2326 Lab1")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/scratch/guszarzmo/lt2326_labs/lab1/data",
                        help="Directory contains h5 files.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/scratch/guszarzmo/lt2326_labs/lab1/checkpoints",  # noqa
        help="Directory to save models duriing training")

    parser.add_argument("--model",
                        type=str,
                        default="2",
                        help="Model to train. 1 or 2")

    parser.add_argument("--load_model",
                        type=str,
                        default="checkpoint.pt",
                        help="Checkpoint filename to load")

    parser.add_argument(
        "--plot_env_name",
        type=str,
        default="lab1_model2_1e-5_plateau",
        help="Visdom env. name to plot the training and validation loss.")

    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU device to be used")

    args = parser.parse_args()

    return parser, args


def vis_init(vis) -> None:
    legend = [["Train", "Validation"], ["Validation"]]
    title = []
    xlabel = []
    ylabel = []
    win_name = []
    for metric in ["Loss", "F1_score"]:
        title.append(f"{metric} Plot")
        xlabel.append("Epoch")
        ylabel.append("Loss")
        win_name.append(f"{metric}_win")
    opt_win = {
        "win_name": win_name,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "title": title,
        "legend": legend
    }
    vis.add_wins(**opt_win)


def vis_plot(vis, y, x, names, split) -> None:
    for xi, yi, name in zip(x, y, names):
        vis.plot_line(yi, xi, split, f"{name}_win")
        vis.vis.save([vis.env_name])


class Train():
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 device,
                 transform,
                 scheduler=None) -> None:

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.transform = transform
        self.scheduler = scheduler

        # metrics
        self.loss = []
        self.epoch_loss = []
        self.val_loss = []
        # self.f1 = []
        # self.epoch_f1 = []
        self.val_f1 = []
        self.best_metric = -1  # best metric is f1 score

    def step(self, images, labels) -> None:
        images = self.transform(images.to(self.device))
        labels = labels.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(images)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()

        # metrics, f1 score is too slow
        # self.f1.append(self.metric(out.detach().cpu(), labels.detach().cpu()))
        self.loss.append(loss.item())

    def eval(self, eval_iter, eval_len) -> None:
        losses = []
        f1s = []
        eval_pb = tqdm(total=eval_len, leave=False, unit="step")

        self.model.eval()
        for step, (images, labels) in enumerate(eval_iter):
            eval_pb.set_description(f"Step: {step:<3d}")

            images = self.transform(images.to(self.device))
            labels = labels.to(self.device)
            with torch.no_grad():
                out = self.model(images)
                loss = self.criterion(out, labels)

            # metrics
            losses.append(loss.item())
            f1s.append(self.metric(out.detach().cpu(), labels.detach().cpu()))
            # update progress bar
            eval_pb.set_postfix({"F1 loss": f1s[-1]})
            eval_pb.update(1)

        # Epoch loss, update progress bar, and plot loss
        val_loss = sum(losses) / len(losses)
        val_f1 = sum(f1s) / len(f1s)
        self.val_loss.append(val_loss)
        self.val_f1.append(val_f1)

    def metric(self, out, targets):
        probs = torch.log_softmax(out, dim=1)
        _, preds = torch.max(probs, dim=1)
        return f1_score(targets.view(-1), preds.view(-1), average="macro")

    def save_checkpoint(self, state_dict, save_dir, best_model=True):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()

        state_dict["model_state_dict"] = model_state
        state_dict["optimizer_state_dict"] = optimizer_state
        state_dict["scheduler_state_dict"] = scheduler_state
        state_dict["best_metric"] = self.best_metric
        model_name = state_dict["model_name"]

        # Save model
        time_tag = str(datetime.now().strftime("%d%m_%Hh%M%S"))
        if best_model:
            filename = f"m{model_name}_{time_tag}.pt"
        else:
            filename = "checkpoint.pt"
        save_path = Path(save_dir) / f"model{model_name}" / filename
        torch.save(state_dict, save_path)


if __name__ == "__main__":

    # parse argument command
    parser, args = parse_arguments()

    # select a device
    device = select_device(args.gpu)

    # Init visualization
    vis = None
    if args.plot_env_name:
        env_name = args.plot_env_name
        vis = Visualizations(env_name=env_name)
        vis_init(vis)

    # intialize some parameters
    lr_init = 1e-5  # change learning rate by adding 5e-6
    lr_factor = 0.75 * lr_init
    batch_size = 16

    # load datasets
    print("data loading ...")
    train_path = str(Path(args.dataset_dir) / "train.h5")
    val_path = str(Path(args.dataset_dir) / "val.h5")
    train_ds = HDF5Dataset(train_path)
    val_ds = HDF5Dataset(val_path)

    pin_memory = device.type == "cuda"
    kwargs = dict(shuffle=True, num_workers=4, pin_memory=pin_memory)
    train_iter = DataLoader(train_ds, batch_size=batch_size, **kwargs)
    val_iter = DataLoader(val_ds, batch_size=1, **kwargs)

    print("Model loading ...\n")
    # construct model, optmizer
    if args.model == "1":
        model = Model_1().to(device)
    elif args.model == "2":
        model = Model_2().to(device)
    else:
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = IncreaseLROnPlateau(optimizer=optimizer, factor=lr_factor)

    # load prev training data
    epoch_start = 0
    best_metric = -1
    val_loss = 0.0
    val_f1 = 0.0
    loss = 0.0
    # f1 = 0.0
    if args.load_model:
        load_dir = Path(args.checkpoint_dir) / f"model{args.model}"
        load_path = load_dir / args.load_model
        model_data = torch.load(load_path, map_location=torch.device("cpu"))
        model_state_dict = model_data["model_state_dict"]
        optimizer_state_dict = model_data["optimizer_state_dict"]
        scheduler_state_dict = model_data["scheduler_state_dict"]

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(scheduler_state_dict)

        # prev. training status
        epoch_start = model_data["epoch"] + 1
        val_loss = model_data["val_loss"]
        val_f1 = model_data["val_f1"]
        best_metric = model_data["best_metric"]
        loss = model_data["loss"]
        # f1 = model_data["f1"]

    # train/val loop: Validation every 2 epochs
    # Construct train class
    transform = transforms.Normalize(*train_ds.mean_stddev)  # normalization
    epochs = 500
    train = Train(model, optimizer, criterion, device, transform, lr_scheduler)
    train.best_metric = best_metric  # load prev. training info
    if val_loss:
        train.val_loss.append(val_loss)
        train.val_f1.append(val_f1)

    # Progress bar setup
    total_steps = (epochs - epoch_start) * ((len(train_ds) // batch_size) + 1)
    train_pb = tqdm(total=total_steps, unit="Step")  # progress bar
    pb_post = {"Step Loss": 0.0, "Epoch Loss": loss, "Eval f1": val_f1}
    train_pb.set_postfix(pb_post)

    for epoch in range(epoch_start, epochs):
        for step, (images, labels) in enumerate(train_iter):
            train.step(images, labels)  # train step

            # update progress bar
            train_pb.set_description(f"Epoch/Step: {epoch:>3d}/{step:<3d}")
            pb_post["Step Loss"] = train.loss[-1]
            train_pb.set_postfix(pb_post)
            train_pb.update(1)

        # eval every 2 epochs
        if (epoch + 1) % 2 == 0:
            train.eval(val_iter, len(val_ds))
            train.scheduler.step(float(train.val_f1[-1]))

            # update progress bar and plot
            pb_post["Eval f1"] = train.val_f1[-1]
            train_pb.set_postfix(pb_post)
            vis_plot(vis, [train.val_loss[-1], train.val_f1[-1]],
                     [epoch, epoch], ["Loss", "F1_score"], "Validation")

        # Epoch loss, update progress bar, and plot loss
        epoch_loss = sum(train.loss) / len(train.loss)
        train.epoch_loss.append(epoch_loss)
        train.loss = []  # reset epoch metrics
        # epoch_f1 = sum(train.f1) / len(train.f1)
        # train.epoch_f1.append(epoch_f1)
        # train.f1 = []  # reset epoch metrics
        pb_post["Epoch Loss"] = epoch_loss
        train_pb.set_postfix(pb_post)

        # save checkpoint/bestmodel
        state_dict = {
            "model_name": args.model,
            "steps": step,
            "epoch": epoch,
            "loss": train.epoch_loss[-1],
            "val_loss": train.val_loss[-1] if train.val_loss else 0,
            # "f1": train.epoch_f1[-1],
            "val_f1": train.val_f1[-1] if train.val_f1 else -1,
        }
        save_dir = args.checkpoint_dir

        if train.val_f1 and train.val_f1[-1] > train.best_metric:
            train.best_metric = train.val_f1[-1]
            train.save_checkpoint(state_dict, save_dir, best_model=True)
        else:
            train.save_checkpoint(state_dict, save_dir, best_model=False)

        # plot loss
        vis_plot(vis, [train.epoch_loss[-1]], [epoch], ["Loss"], "Train")
