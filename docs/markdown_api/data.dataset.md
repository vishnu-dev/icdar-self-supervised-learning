<!-- markdownlint-disable -->

<a href="../../src/data/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.dataset`






---

<a href="../../src/data/dataset.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ICDARDataset`




<a href="../../src/data/dataset.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    csv_filepath,
    root_dir,
    transforms=None,
    convert_rgb=True,
    mask_generator=None
)
```

ICDAR Custom Dataset class 



**Args:**
 
 - <b>`csv_filepath`</b> (str):  Label file path 
 - <b>`root_dir`</b> (str):  Dataset root directory 
 - <b>`transforms`</b> (torchvision.transforms, optional):  Transforms. Defaults to None. 
 - <b>`convert_rgb`</b> (bool, optional):  Whether to convert to RGB. Defaults to True. 
 - <b>`mask_generator`</b> (Union[Callable, NoneType], optional):  Mask generator. Defaults to None. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
