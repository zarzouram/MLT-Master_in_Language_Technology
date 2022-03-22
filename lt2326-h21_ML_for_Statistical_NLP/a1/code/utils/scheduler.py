"""This code is based on the pytorch impelementation of ReduceLROnPlateau
scheduler.
https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
"""
# %%
# from torch.optim.lr_scheduler import _LRScheduler
from math import inf


class IncreaseLROnPlateau(object):
    """Increase learning rate when a metric has stopped improving.  This
    scheduler reads a metrics quantity and if the quantity monitored has
    stopped increasing for a 'patience' number of epochs, the learning rate is
    increased.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        factor (float): Factor by which the learning rate will be
            increased. new_lr = lr + factor.

        patience (int): Number of epochs with no improvement after
            which learning rate will be increased. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.

        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.

        min_lr (float): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.

        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self,
                 optimizer,
                 factor=5e-6,
                 patience=2,
                 threshold=1e-4,
                 max_lr=1e-3,
                 verbose=False):

        self.optimizer = optimizer
        self.max_lr = [max_lr] * len(optimizer.param_groups)

        self.verbose = verbose

        self.factor = factor
        self.worse = -inf  # the worse value
        self.best = -inf

        self.threshold = threshold
        self.patience = patience

        self.num_bad_epochs = 0
        self.last_epoch = 0

    def step(self, metrics: float):

        self.last_epoch += 1

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.num_bad_epochs = 0

        self.optimizer._last_lr = [
            group['lr'] for group in self.optimizer.param_groups
        ]

    def is_better(self, value, best_value):
        return value > best_value + self.threshold

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(old_lr + self.factor, self.max_lr[i])
            param_group['lr'] = new_lr
            if self.verbose:
                print('\tEpoch {:3d}: reducing learning rate'
                      ' of group {} from {} to {}.'.format(
                          epoch, i, old_lr, new_lr))

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


# %%

# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim

# p = nn.Parameter(torch.empty(4, 4))
# optimizer = optim.Adam([p], lr=0.5)
# lr_scheduler = IncreaseLROnPlateau(optimizer=optimizer,
#                                    factor=0.1,
#                                    verbose=True)

# # Plotting
# plt.figure(figsize=(15, 15))

# lrs = []
# metrics_fake = list(range(1, 6)) + [10] * 5 + list(range(
#     11, 16)) + [15 - n * 1 for n in range(1, 6)] + list(range(11, 16))
# for metric in metrics_fake:
#     optimizer.step()
#     lr_scheduler.step(metric)
#     lrs.extend([group['lr'] for group in optimizer.param_groups])

# plt.plot(range(len(lrs)), lrs)
# plt.xticks(range(len(lrs) + 1))
# plt.ylabel("Learning rate")
# plt.xlabel("Iterations (in batches)")
# plt.title("Noam Learning Rate Scheduler")
# plt.show()
# print(len(metrics_fake))

# %%
