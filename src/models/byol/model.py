#-----------------------------------------------------------------------------------------
# Bootstrap your own latent (BYOL) model by DeepMind with modifications
# and a port to PyTorch Lightning
#
# Reference: https://github.com/lucidrains/byol-pytorch
# 
# @misc{grill2020bootstrap,
#   title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
#   author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
#   year = {2020},
#   eprint = {2006.07733},
#   archivePrefix = {arXiv},
#   primaryClass = {cs.LG}
# }
#-----------------------------------------------------------------------------------------

import copy
import random
from functools import wraps

import math
import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms as T
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    """Singleton pattern decorator

    Args:
        cache_key (str): Cache key
    """
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    """Negative cosine similarity loss as defined in the paper"""
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class EMA():
    def __init__(self, beta):
        """Exponential moving average

        Args:
            beta (float): Exponential decay
        """
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def MLP(dim, projection_size, hidden_size=4096):
    """Simple MLP with ReLU activation and batch norm

    Args:
        dim (int): Input dimensions
        projection_size (int): Projection size
        hidden_size (int, optional): Hidden dimensions. Defaults to 4096.

    Returns:
        torch.nn.Sequential: MLP
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096):
    """SimSiam MLP with ReLU activation and batch norm
    
    Args:
        dim (int): Input dimensions
        projection_size (int): Projection size
        Hidden_size (int, optional): Hidden dimensions. Defaults to 4096.
    
    Returns:
        torch.nn.Sequential: SimSiam MLP
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):
        """Wrapper for backbone network with a projection head

        Args:
            net (torch.nn.Module): Backbone network
            projection_size (int): Projection size
            projection_hidden_size (int): Projection hidden dimensions
            layer (int, optional): Layer index to find. Defaults to -2.
            use_simsiam_mlp (bool, optional): Whether to use SimSiam MLP. Defaults to False.
        """
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(pl.LightningModule):
    """Bootstrap your own latent (BYOL) model
    """
    def __init__(
        self,
        image_size=224,
        base_encoder = "resnet50",
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        first_conv: bool = True,
        maxpool1: bool = True,
        **kwargs
    ):
        """BYOL Initialization

        Args:
            image_size (int, optional): Image size. Defaults to 224.
            base_encoder (str, optional): Backbone model architecture to use. Defaults to "resnet50".
            hidden_layer (int, optional): Hidden layer index to find. Defaults to -2.
            projection_size (int, optional): Projection size. Defaults to 256.
            projection_hidden_size (int, optional): Projection hidden dimensions. Defaults to 4096.
            moving_average_decay (float, optional): Moving average decay. Defaults to 0.99.
            use_momentum (bool, optional): Whether to use momentum for the target encoder. Defaults to True.
            first_conv (bool, optional): Whether to use first conv layer. Defaults to True.
            maxpool1 (bool, optional): Whether to use maxpool1. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")
        
        self.model_kwargs = kwargs
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.arch = base_encoder
        
        self.net = self.init_model()

        self.online_encoder = NetWrapper(self.net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # send a mock image tensor to instantiate singleton parameters
        batch = torch.randn(2, 3, image_size, image_size, device=self.device), torch.randn(2, 3, image_size, image_size, device=self.device), None
        self.forward(batch)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        else:
            raise NotImplementedError(f"Backbone {self.arch} not implemented")

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        batch,
        return_embedding = False,
        return_projection = True
    ):

        image_one, image_two, _ = batch

        # eval mode returns projections and embeddings
        if return_embedding:
            return self.online_encoder(image_one, return_projection = return_projection)

        assert not (self.training and image_one.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # forward pass of online encoder
        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        # forward pass of target encoder
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        # compute loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
    
    def training_step(self, batch, batch_idx):

        images, _ = batch

        loss = self.forward(images, return_embedding=False, return_projection=False)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        for parameters in self.parameters():
            if parameters.grad is not None:
                grad_norm = torch.norm(parameters.grad)
                self.logger.experiment.add_scalar('grad_norm', grad_norm, self.global_step)

        self.log('train_loss', loss, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        images, _ = batch
                
        loss = self.forward(images, return_embedding=False, return_projection=False)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping validation".format(loss_value))
            sys.exit(1)

        self.log('val_loss', loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        
        # Adam optimizer with weight decay
        optimizer = Adam(
            self.parameters(),
            lr=self.model_kwargs['learning_rate'],
            weight_decay=self.model_kwargs['weight_decay']
        )
        
        # Cosine annealing with linear warmup for learning rate
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.model_kwargs['warmup_epochs'],
            max_epochs=self.model_kwargs['max_epochs'],
        )
        
        return [optimizer], [scheduler]
    
    def on_before_zero_grad(self, _):
        
        # update moving average of target encoder
        if self.use_momentum:
            self.update_moving_average()