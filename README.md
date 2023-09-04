<!-- markdownlint-disable -->
# Comparison of Self-Supervised Learning models for ICDAR CLaMM Challenge

This project presents a comparison of self-supervised learning methods
for different downstream tasks in the context of Medieval Handwriting in
the Latin Script dataset. Self-supervised learning has shown promise in
various computer vision and natural language processing applications,
but its effectiveness on historical scripts has not been extensively
explored. 

<img src="docs/Samples of CLaMM Dataset.jpg"/>

Three self-supervised learning methods are compared in this work.
* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), 
* [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
* [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)


The performance evaluation was
conducted on one downstream tasks i.e. script type classification. The
results indicate that the SimCLR method outperforms other methods in the
downstream task for the Medieval Handwritings Script dataset.
Additionally, insights were provided regarding the factors influencing
the performance of self-supervised learning methods in this context,
including the selection of pre-training data and the size of the
pre-training dataset. In conclusion, this study showcases the potential
of self-supervised learning for historical handwritten document
classification tasks and emphasizes the significance of selecting
suitable methods for specific downstream tasks.

## Dataset

ICDAR CLaMM Challenge dataset is used for this project. The dataset can be found [here](https://clamm.irht.cnrs.fr/icdar-2017/download/)

## Documentation

API Documentation is available at [DOCUMENTATION.md](./DOCUMENTATION.md)

## Running the code

### Prerequisites

`pip install -r requirements.txt`

### Training

#### SSL Model Training

```bash
cd src/
python train.py +experiment=simclr_bolts
```

#### Linear Classifier Training

```bash
cd src/
python evaluate.py +experiment=simclr_eval
```

### Evaluation

#### Linear Classifier Testing

Check notebook [here](./notebooks/downstream_tasks.ipynb)

## Results

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">Pre-training</th>
    <th class="tg-c3ow" colspan="2">Linear evaluation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Model Name</td>
    <td class="tg-c3ow">Epochs</td>
    <td class="tg-c3ow">Batch size</td>
    <td class="tg-c3ow">Training epochs</td>
    <td class="tg-c3ow">Top-1 accuracy</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SimCLR</td>
    <td class="tg-c3ow">500</td>
    <td class="tg-c3ow">256</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">71.8 %</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MAE</td>
    <td class="tg-c3ow">500</td>
    <td class="tg-c3ow">256</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">36.1 %</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BYOL</td>
    <td class="tg-c3ow">500</td>
    <td class="tg-c3ow">64</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">45.2 %</td>
  </tr>
</tbody>
</table>

Image sources: [ICDAR CLaMM](https://clamm.irht.cnrs.fr/script-classes/)
