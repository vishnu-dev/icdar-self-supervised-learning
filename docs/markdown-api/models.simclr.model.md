<!-- markdownlint-disable -->

<a href="../../src/models/simclr/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.simclr.model`






---

<a href="../../src/models/simclr/model.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimCLR`




<a href="../../src/models/simclr/model.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(batch_size, num_samples, max_epoch)
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

<a href="../../src/models/simclr/model.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../../src/models/simclr/model.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```





---

<a href="../../src/models/simclr/model.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `training_step`

```python
training_step(train_batch, batch_idx)
```





---

<a href="../../src/models/simclr/model.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validation_step`

```python
validation_step(val_batch, batch_idx)
```






