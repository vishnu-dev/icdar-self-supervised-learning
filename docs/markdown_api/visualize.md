<!-- markdownlint-disable -->

<a href="../../src/visualize.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualize`





---

<a href="../../src/visualize.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_features`

```python
plot_features(model, data_loader, num_feats, batch_size, num_samples)
```

Plot embeddings. This is a wrapper around : func : ` tsne. TSNE ` to make it easier to visualize the model's performance. 



**Args:**
 
 - <b>`model`</b>:  The model to be visualized. It must have a ` eval ` method that takes a list of inputs and returns a 2 - tuple ( x1 data_loader num_feats batch_size num_samples 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
