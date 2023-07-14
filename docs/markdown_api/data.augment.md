<!-- markdownlint-disable -->

<a href="../../src/data/augment.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.augment`






---

<a href="../../src/data/augment.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GaussianNoise`




<a href="../../src/data/augment.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(mean=0.0, std=1.0)
```

Gaussian noise transform. 



**Args:**
 
 - <b>`mean`</b> (float, optional):  Mean of the distribution for noise. Defaults to 0.0. 
 - <b>`std`</b> (_type_, optional):  Standard deviation of the distribution for noise. Defaults to 1.0. 





---

<a href="../../src/data/augment.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Erosion`




<a href="../../src/data/augment.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(kernel_size=3)
```

Erosion transform. 



**Args:**
 
 - <b>`kernel_size`</b> (int, optional):  Kernel size for erosion. Defaults to 3. 





---

<a href="../../src/data/augment.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dilation`




<a href="../../src/data/augment.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(kernel_size=3)
```

Dilation transform. 



**Args:**
 
 - <b>`kernel_size`</b> (int, optional):  Kernel size for dilation. Defaults to 3. 





---

<a href="../../src/data/augment.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PairTransform`




<a href="../../src/data/augment.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(transforms=None, online_transforms=None)
```

Pair transform for contrastive models. 



**Args:**
 
 - <b>`transforms`</b> (torchvision.transforms, optional):  Transforms for the pair. Defaults to None. 
 - <b>`online_transforms`</b> (torchvision.transforms, optional):  Transforms for the online image. Defaults to None. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
