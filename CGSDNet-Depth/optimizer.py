import math
from typing import Union, Iterable

from torch import optim, Tensor


class PolyScheduler(object):
    def __init__(self,
                 optimizer: optim.Optimizer,
                 total_epochs: int,
                 epoch_steps: int,
                 lr_power: float = 0.9,
                 use_warmup: bool = False,
                 warmup_epochs: int = 5,
                 warmup_factor: float = 1e-3,
                 scheduler_type: str = 'batch'):
        assert epoch_steps > 0 and total_epochs > 0
        assert scheduler_type == 'batch' or scheduler_type == 'epoch'
        if use_warmup is False:
            warmup_epochs = 0

        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.total_epochs = total_epochs
        self.lr_power = lr_power
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.scheduler_type = scheduler_type

        self.warmup_steps = self.warmup_epochs * self.epoch_steps
        self.leftover_steps = (self.total_epochs - self.warmup_epochs) * self.epoch_steps

    def get_batch_poly_lr(self,
                          step: int) -> float:
        if self.use_warmup is True and step <= self.warmup_steps:
            alpha = float(step) / self.warmup_steps
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            current_step = step - self.warmup_steps
            leftover_steps = self.leftover_steps
            poly_lr = math.pow(1 - current_step / leftover_steps, self.lr_power)
            return poly_lr

    def get_epoch_poly_lr(self,
                          epoch: int) -> float:
        if self.use_warmup is True:
            if epoch <= self.warmup_epochs:
                alpha = (float(epoch) + 1) / (self.warmup_epochs + 1)
                return self.warmup_factor * (1 - alpha) + alpha
            else:
                poly_lr = math.pow(1 - (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                                   self.lr_power)
                return poly_lr
        else:
            poly_lr = math.pow(1 - epoch / self.total_epochs, self.lr_power)
            return poly_lr

    def get_poly_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        if self.scheduler_type == 'batch':
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.get_batch_poly_lr)
        elif self.scheduler_type == 'epoch':
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.get_epoch_poly_lr)
        else:
            raise ValueError('No such scheduler type!')
        return scheduler


class NullScheduler(object):
    def __init__(self,
                 optimizer: optim.Optimizer) -> None:
        self.optimizer = optimizer

    def get_null_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1)
        return scheduler


class StepScheduler(object):
    def __init__(self,
                 optimizer: optim.Optimizer,
                 step_size: int,
                 gamma: float = 0.1) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

    def get_step_scheduler(self) -> optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        return scheduler


# SGD optimizer
class SGD(optim.SGD):
    def __init__(self,
                 params: Union[Iterable[Tensor], Iterable[dict]],
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 dampening: float = 0,
                 weight_decay: float = 0.0005) -> None:
        super(SGD, self).__init__(params=params, lr=lr, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay)


# AdamW optimizer
class AdamW(optim.AdamW):
    def __init__(self,
                 params: Union[Iterable[Tensor], Iterable[dict]],
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4) -> None:
        super(AdamW, self).__init__(params=params, lr=lr, weight_decay=weight_decay)
