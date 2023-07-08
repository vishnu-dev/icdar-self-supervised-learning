import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class CustomScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, min_lr, lr):
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.lr = lr
        super(CustomScheduler, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs 
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return [lr] * len(self.optimizer.param_groups)