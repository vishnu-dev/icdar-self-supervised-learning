<!-- markdownlint-disable -->
# Comparison of Self-Supervised Learning models for ICDAR CLaMM Challenge

This project presents a comparison of self-supervised learning methods
for different downstream tasks in the context of Medieval Handwriting in
the Latin Script dataset. Self-supervised learning has shown promise in
various computer vision and natural language processing applications,
but its effectiveness on historical scripts has not been extensively
explored. 


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
