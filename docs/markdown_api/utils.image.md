<!-- markdownlint-disable -->

<a href="../../src/utils/image.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils.image`





---

<a href="../../src/utils/image.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `img_is_color`

```python
img_is_color(img)
```

Check if an image is color or grayscale. 



**Args:**
 
 - <b>`img`</b> (PIL.Image):  Image to be checked. 



**Returns:**
 
 - <b>`bool`</b>:  If True, the image is color. If False, the image is grayscale. 


---

<a href="../../src/utils/image.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `show_image_list`

```python
show_image_list(
    list_images,
    list_titles=None,
    list_cmaps=None,
    grid=True,
    num_cols=2,
    figsize=(40, 40),
    title_fontsize=30
)
```

Shows a grid of images, where each image is a Numpy array. The images can be either RGB or grayscale. 



**Args:**
 
 - <b>`images`</b> (list):  List of the images to be displayed. 
 - <b>`list_titles`</b> (list or None):  Optional list of titles to be shown for each image. 
 - <b>`list_cmaps`</b> (list or None):  Optional list of cmap values for each image. If None, then cmap will be  automatically inferred. 
 - <b>`grid`</b> (bool):  If True, show a grid over each image. 
 - <b>`num_cols`</b> (int):  Number of columns to show. 
 - <b>`figsize`</b> (tuple):  Tuple of width and height, value to be passed to pyplot.figure(). 
 - <b>`title_fontsize`</b> (int):  Value to be passed to set_title(). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
