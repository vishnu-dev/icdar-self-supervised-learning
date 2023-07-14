<!-- markdownlint-disable -->

<a href="../../src/train.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `train`





---

<a href="../../train/execute#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `execute`

```python
execute(cfg: DictConfig)
```

Configuration based model training entry point. CLI arguments are passed as configuration overrides. 



**Args:**
 
 - <b>`cfg`</b>:  The configuration object from hydra. 



**Examples:**
 ``` python train.py +experiment=simclr_bolts model.params.batch_size=128```





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
