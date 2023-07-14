<!-- markdownlint-disable -->

<a href="../../src/models/byol/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.byol.model`





---

<a href="../../src/models/byol/model.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `default`

```python
default(val, def_val)
```






---

<a href="../../src/models/byol/model.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flatten`

```python
flatten(t)
```






---

<a href="../../src/models/byol/model.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `singleton`

```python
singleton(cache_key)
```

Singleton pattern decorator 



**Args:**
 
 - <b>`cache_key`</b> (str):  Cache key 


---

<a href="../../src/models/byol/model.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_module_device`

```python
get_module_device(module)
```






---

<a href="../../src/models/byol/model.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_requires_grad`

```python
set_requires_grad(model, val)
```






---

<a href="../../src/models/byol/model.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loss_fn`

```python
loss_fn(x, y)
```

Negative cosine similarity loss as defined in the paper 


---

<a href="../../src/models/byol/model.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_moving_average`

```python
update_moving_average(ema_updater, ma_model, current_model)
```






---

<a href="../../src/models/byol/model.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `MLP`

```python
MLP(dim, projection_size, hidden_size=4096)
```

Simple MLP with ReLU activation and batch norm 



**Args:**
 
 - <b>`dim`</b> (int):  Input dimensions 
 - <b>`projection_size`</b> (int):  Projection size 
 - <b>`hidden_size`</b> (int, optional):  Hidden dimensions. Defaults to 4096. 



**Returns:**
 
 - <b>`torch.nn.Sequential`</b>:  MLP 


---

<a href="../../src/models/byol/model.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SimSiamMLP`

```python
SimSiamMLP(dim, projection_size, hidden_size=4096)
```

SimSiam MLP with ReLU activation and batch norm 



**Args:**
 
 - <b>`dim`</b> (int):  Input dimensions 
 - <b>`projection_size`</b> (int):  Projection size 
 - <b>`Hidden_size`</b> (int, optional):  Hidden dimensions. Defaults to 4096. 



**Returns:**
 
 - <b>`torch.nn.Sequential`</b>:  SimSiam MLP 


---

<a href="../../src/models/byol/model.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EMA`




<a href="../../src/models/byol/model.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta)
```

Exponential moving average 



**Args:**
 
 - <b>`beta`</b> (float):  Exponential decay 




---

<a href="../../src/models/byol/model.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_average`

```python
update_average(old, new)
```






---

<a href="../../src/models/byol/model.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NetWrapper`




<a href="../../src/models/byol/model.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    net,
    projection_size,
    projection_hidden_size,
    layer=-2,
    use_simsiam_mlp=False
)
```

Wrapper for backbone network with a projection head 



**Args:**
 
 - <b>`net`</b> (torch.nn.Module):  Backbone network 
 - <b>`projection_size`</b> (int):  Projection size 
 - <b>`projection_hidden_size`</b> (int):  Projection hidden dimensions 
 - <b>`layer`</b> (int, optional):  Layer index to find. Defaults to -2. 
 - <b>`use_simsiam_mlp`</b> (bool, optional):  Whether to use SimSiam MLP. Defaults to False. 




---

<a href="../../src/models/byol/model.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, return_projection=True)
```





---

<a href="../../src/models/byol/model.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_representation`

```python
get_representation(x)
```






---

<a href="../../src/models/byol/model.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BYOL`




<a href="../../src/models/byol/model.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    image_size=224,
    base_encoder='resnet50',
    hidden_layer=-2,
    projection_size=256,
    projection_hidden_size=4096,
    moving_average_decay=0.99,
    use_momentum=True,
    first_conv: bool = True,
    maxpool1: bool = True,
    **kwargs
)
```

Bootstrap your own latent (BYOL) model 



**Args:**
 
 - <b>`image_size`</b> (int, optional):  Image size. Defaults to 224. 
 - <b>`base_encoder`</b> (str, optional):  Backbone model architecture to use. Defaults to "resnet50". 
 - <b>`hidden_layer`</b> (int, optional):  Hidden layer index to find. Defaults to -2. 
 - <b>`projection_size`</b> (int, optional):  Projection size. Defaults to 256. 
 - <b>`projection_hidden_size`</b> (int, optional):  Projection hidden dimensions. Defaults to 4096. 
 - <b>`moving_average_decay`</b> (float, optional):  Moving average decay. Defaults to 0.99. 
 - <b>`use_momentum`</b> (bool, optional):  Whether to use momentum for the target encoder. Defaults to True. 
 - <b>`first_conv`</b> (bool, optional):  Whether to use first conv layer. Defaults to True. 
 - <b>`maxpool1`</b> (bool, optional):  Whether to use maxpool1. Defaults to True. 


---

#### <kbd>property</kbd> automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> dtype





---

#### <kbd>property</kbd> example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> trainer





---

#### <kbd>property</kbd> truncated_bptt_steps

Enables `Truncated Backpropagation Through Time` in the Trainer when set to a positive integer. 

It represents the number of times :meth:`training_step` gets called before backpropagation. If this is > 0, the :meth:`training_step` receives an additional argument ``hiddens`` and is expected to return a hidden state. 

---

#### <kbd>property</kbd> use_amp







---

<a href="../../src/models/byol/model.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../../src/models/byol/model.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch, return_embedding=False, return_projection=True)
```





---

<a href="../../src/models/byol/model.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_model`

```python
init_model()
```





---

<a href="../../src/models/byol/model.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_before_zero_grad`

```python
on_before_zero_grad(_)
```





---

<a href="../../src/models/byol/model.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset_moving_average`

```python
reset_moving_average()
```





---

<a href="../../src/models/byol/model.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `training_step`

```python
training_step(batch, batch_idx)
```





---

<a href="../../src/models/byol/model.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_moving_average`

```python
update_moving_average()
```





---

<a href="../../src/models/byol/model.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validation_step`

```python
validation_step(batch, batch_idx)
```






