<!-- markdownlint-disable -->

<a href="../../src/models/mae/lr_sched.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.mae.lr_sched`






---

<a href="../../src/models/mae/lr_sched.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CustomScheduler`




<a href="../../src/models/mae/lr_sched.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(optimizer, warmup_epochs, epochs, min_lr, lr)
```

Custom learning rate scheduler Half-cycle cosine learning rate scheduler with warmup epochs 



**Args:**
 
 - <b>`optimizer`</b> (torch.optim):  Optimizer 
 - <b>`warmup_epochs`</b> (int):  Number of warmup epochs 
 - <b>`epochs`</b> (int):  Number of epochs 
 - <b>`min_lr`</b> (float):  Minimum learning rate 
 - <b>`lr`</b> (float):  Base learning rate 




---

<a href="../../src/models/mae/lr_sched.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_lr`

```python
get_lr()
```






