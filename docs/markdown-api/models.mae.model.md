<!-- markdownlint-disable -->

<a href="../../src/models/mae/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.mae.model`






---

<a href="../../src/models/mae/model.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MAE`
Masked Autoencoder with VisionTransformer backbone  



<a href="../../src/models/mae/model.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.0,
    norm_layer=<class 'torch.nn.modules.normalization.LayerNorm'>,
    norm_pix_loss=False,
    **kwargs
)
```






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

<a href="../../src/models/mae/model.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `configure_optimizers`

```python
configure_optimizers() → Any
```





---

<a href="../../src/models/mae/model.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(imgs, mask_ratio=0.75)
```





---

<a href="../../src/models/mae/model.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_decoder`

```python
forward_decoder(x, ids_restore)
```





---

<a href="../../src/models/mae/model.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_encoder`

```python
forward_encoder(x, mask_ratio)
```





---

<a href="../../src/models/mae/model.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_loss`

```python
forward_loss(imgs, pred, mask)
```

imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,  

---

<a href="../../src/models/mae/model.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `initialize_weights`

```python
initialize_weights()
```





---

<a href="../../src/models/mae/model.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `patchify`

```python
patchify(imgs)
```

imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3) 

---

<a href="../../src/models/mae/model.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `random_masking`

```python
random_masking(x, mask_ratio)
```

Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random noise. x: [N, L, D], sequence 

---

<a href="../../src/models/mae/model.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `training_step`

```python
training_step(batch, batch_idx)
```





---

<a href="../../src/models/mae/model.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `unpatchify`

```python
unpatchify(x)
```

x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W) 

---

<a href="../../src/models/mae/model.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validation_step`

```python
validation_step(batch, batch_idx)
```






