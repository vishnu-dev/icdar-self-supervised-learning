<!-- markdownlint-disable -->

<a href="../../src/pipeline/lightning.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.lightning`






---

<a href="../../src/pipeline/lightning.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LightningPipeline`




<a href="../../src/pipeline/lightning.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(root_dir, model_class, mode, data_loader, batch_size, trainer_cfg)
```

Pytorch Lightning pipeline 



**Args:**
 
 - <b>`root_dir`</b> (str):  Root directory for model checkpoints 
 - <b>`model_class`</b> (Any):  Model class 
 - <b>`mode`</b> (str):  Execution mode (train, eval, test) 
 - <b>`data_loader`</b> (Dict):  Data loader dictionary 
 - <b>`batch_size`</b> (int):  Batch size 
 - <b>`trainer_cfg`</b> (Dict):  Dictionary of trainer configuration parameters 




---

<a href="../../src/pipeline/lightning.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_model`

```python
init_model()
```





---

<a href="../../src/pipeline/lightning.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_trainer`

```python
init_trainer(model)
```





---

<a href="../../src/pipeline/lightning.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
