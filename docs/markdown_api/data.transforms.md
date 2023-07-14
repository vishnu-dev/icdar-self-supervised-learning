<!-- markdownlint-disable -->

<a href="../../src/data/transforms.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.transforms`





---

<a href="../../src/data/transforms.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `transform_factory`

```python
transform_factory(model_name, mode, config=None)
```

Transform factory for self-supervised models 



**Args:**
 
 - <b>`model_name`</b> (str):  Name of the model 
 - <b>`mode`</b> (str):  Execution mode (train, test) 
 - <b>`config`</b> (dict, optional):  Configuration parameters from hydra. Defaults to None. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If transform is not implemented for the given model or mode 



**Returns:**
 
 - <b>`torchvision.transforms.Compose`</b>:  Transforms 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
