<!-- markdownlint-disable -->

<a href="../../src/data/data_factory.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.data_factory`





---

<a href="../../src/data/data_factory.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `data_factory`

```python
data_factory(
    dataset_name,
    root_dir,
    label_filepath,
    train_val_test_ratio,
    transforms,
    mode,
    batch_size,
    collate_fn=None,
    num_cpus=None,
    pin_memory=True
)
```

Data loader factory based on dataset name. 



**Args:**
 
 - <b>`dataset_name`</b> (str):  Name of the dataset 
 - <b>`root_dir`</b> (str):  Dataset root directory 
 - <b>`label_filepath`</b> (str):  Label CSV filepath 
 - <b>`train_val_test_ratio`</b> (List[float]):  List of ratios for train, val and test 
 - <b>`transforms`</b> (torchvision.transforms.Compose):  Transforms to apply to the dataset 
 - <b>`mode`</b> (str):  Execution mode (train, test) 
 - <b>`batch_size`</b> (int):  Batch size 
 - <b>`collate_fn`</b> (Union[Callable, NoneType], optional):  Collate function. Defaults to None. 
 - <b>`num_cpus`</b> (int, optional):  Number of CPUs for data loading. Defaults to None. 
 - <b>`pin_memory`</b> (bool, optional):  Whether to pin memory. Defaults to True. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If dataset is not implemented 
 - <b>`KeyError`</b>:  If defined mode is not implemented 



**Returns:**
 
 - <b>`dict[str, torch.utils.data.DataLoader]`</b>:  Dictionary of data loaders. Train and val for train mode, test for test mode 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
