<!-- markdownlint-disable -->

# API Overview

## Modules

- [`data`](./docs/markdown_api/data.md#module-data)
- [`data.augment`](./docs/markdown_api/data.augment.md#module-dataaugment)
- [`data.collate`](./docs/markdown_api/data.collate.md#module-datacollate)
- [`data.data_factory`](./docs/markdown_api/data.data_factory.md#module-datadata_factory)
- [`data.dataset`](./docs/markdown_api/data.dataset.md#module-datadataset)
- [`data.transforms`](./docs/markdown_api/data.transforms.md#module-datatransforms)
- [`eval`](./docs/markdown_api/eval.md#module-eval)
- [`hp_optim`](./docs/markdown_api/hp_optim.md#module-hp_optim)
- [`models`](./docs/markdown_api/models.md#module-models)
- [`models.backbone_factory`](./docs/markdown_api/models.backbone_factory.md#module-modelsbackbone_factory)
- [`models.byol`](./docs/markdown_api/models.byol.md#module-modelsbyol)
- [`models.byol.model`](./docs/markdown_api/models.byol.model.md#module-modelsbyolmodel)
- [`models.mae`](./docs/markdown_api/models.mae.md#module-modelsmae)
- [`models.mae.lr_sched`](./docs/markdown_api/models.mae.lr_sched.md#module-modelsmaelr_sched)
- [`models.mae.model`](./docs/markdown_api/models.mae.model.md#module-modelsmaemodel)
- [`models.mae.pos_embed`](./docs/markdown_api/models.mae.pos_embed.md#module-modelsmaepos_embed)
- [`models.model_factory`](./docs/markdown_api/models.model_factory.md#module-modelsmodel_factory)
- [`models.simclr`](./docs/markdown_api/models.simclr.md#module-modelssimclr)
- [`models.simclr.encoder`](./docs/markdown_api/models.simclr.encoder.md#module-modelssimclrencoder)
- [`models.simclr.head`](./docs/markdown_api/models.simclr.head.md#module-modelssimclrhead)
- [`models.simclr.loss`](./docs/markdown_api/models.simclr.loss.md#module-modelssimclrloss)
- [`models.simclr.model`](./docs/markdown_api/models.simclr.model.md#module-modelssimclrmodel)
- [`pipeline`](./docs/markdown_api/pipeline.md#module-pipeline)
- [`pipeline.callback_factory`](./docs/markdown_api/pipeline.callback_factory.md#module-pipelinecallback_factory)
- [`pipeline.lightning`](./docs/markdown_api/pipeline.lightning.md#module-pipelinelightning)
- [`train`](./docs/markdown_api/train.md#module-train)
- [`utils`](./docs/markdown_api/utils.md#module-utils)
- [`utils.image`](./docs/markdown_api/utils.image.md#module-utilsimage)
- [`visualize`](./docs/markdown_api/visualize.md#module-visualize)

## Classes

- [`augment.Dilation`](./docs/markdown_api/data.augment.md#class-dilation)
- [`augment.Erosion`](./docs/markdown_api/data.augment.md#class-erosion)
- [`augment.GaussianNoise`](./docs/markdown_api/data.augment.md#class-gaussiannoise)
- [`augment.PairTransform`](./docs/markdown_api/data.augment.md#class-pairtransform)
- [`dataset.ICDARDataset`](./docs/markdown_api/data.dataset.md#class-icdardataset)
- [`model.BYOL`](./docs/markdown_api/models.byol.model.md#class-byol)
- [`model.EMA`](./docs/markdown_api/models.byol.model.md#class-ema)
- [`model.NetWrapper`](./docs/markdown_api/models.byol.model.md#class-netwrapper)
- [`lr_sched.CustomScheduler`](./docs/markdown_api/models.mae.lr_sched.md#class-customscheduler)
- [`model.MAE`](./docs/markdown_api/models.mae.model.md#class-mae): Masked Autoencoder with VisionTransformer backbone
- [`encoder.ResNet50Encoder`](./docs/markdown_api/models.simclr.encoder.md#class-resnet50encoder)
- [`head.ProjectionHead`](./docs/markdown_api/models.simclr.head.md#class-projectionhead)
- [`loss.ContrastiveLoss`](./docs/markdown_api/models.simclr.loss.md#class-contrastiveloss)
- [`model.SimCLR`](./docs/markdown_api/models.simclr.model.md#class-simclr)
- [`lightning.LightningPipeline`](./docs/markdown_api/pipeline.lightning.md#class-lightningpipeline)

## Functions

- [`collate.collate_factory`](./docs/markdown_api/data.collate.md#function-collate_factory): Custom collate function for each model.
- [`data_factory.data_factory`](./docs/markdown_api/data.data_factory.md#function-data_factory): Data loader factory based on dataset name.
- [`transforms.transform_factory`](./docs/markdown_api/data.transforms.md#function-transform_factory): Transform factory for self-supervised models
- [`eval.execute`](./docs/markdown_api/eval.md#function-execute): Evaluation entry point.
- [`hp_optim.objective`](./docs/markdown_api/hp_optim.md#function-objective): Objective function for Optuna.
- [`backbone_factory.backbone_factory`](./docs/markdown_api/models.backbone_factory.md#function-backbone_factory): Backbone factory for self-supervised models
- [`model.MLP`](./docs/markdown_api/models.byol.model.md#function-mlp): Simple MLP with ReLU activation and batch norm
- [`model.SimSiamMLP`](./docs/markdown_api/models.byol.model.md#function-simsiammlp): SimSiam MLP with ReLU activation and batch norm
- [`model.default`](./docs/markdown_api/models.byol.model.md#function-default)
- [`model.flatten`](./docs/markdown_api/models.byol.model.md#function-flatten)
- [`model.get_module_device`](./docs/markdown_api/models.byol.model.md#function-get_module_device)
- [`model.loss_fn`](./docs/markdown_api/models.byol.model.md#function-loss_fn): Negative cosine similarity loss as defined in the paper
- [`model.set_requires_grad`](./docs/markdown_api/models.byol.model.md#function-set_requires_grad)
- [`model.singleton`](./docs/markdown_api/models.byol.model.md#function-singleton): Singleton pattern decorator
- [`model.update_moving_average`](./docs/markdown_api/models.byol.model.md#function-update_moving_average)
- [`pos_embed.get_1d_sincos_pos_embed_from_grid`](./docs/markdown_api/models.mae.pos_embed.md#function-get_1d_sincos_pos_embed_from_grid): embed_dim: output dimension for each position
- [`pos_embed.get_2d_sincos_pos_embed`](./docs/markdown_api/models.mae.pos_embed.md#function-get_2d_sincos_pos_embed): grid_size: int of the grid height and width
- [`pos_embed.get_2d_sincos_pos_embed_from_grid`](./docs/markdown_api/models.mae.pos_embed.md#function-get_2d_sincos_pos_embed_from_grid)
- [`pos_embed.interpolate_pos_embed`](./docs/markdown_api/models.mae.pos_embed.md#function-interpolate_pos_embed)
- [`model_factory.model_factory`](./docs/markdown_api/models.model_factory.md#function-model_factory): Model factory for self-supervised models
- [`encoder.test`](./docs/markdown_api/models.simclr.encoder.md#function-test)
- [`loss.test`](./docs/markdown_api/models.simclr.loss.md#function-test)
- [`callback_factory.callback_factory`](./docs/markdown_api/pipeline.callback_factory.md#function-callback_factory): Model callback factory
- [`train.execute`](./docs/markdown_api/train.md#function-execute): Configuration based model training entry point.
- [`image.img_is_color`](./docs/markdown_api/utils.image.md#function-img_is_color): Check if an image is color or grayscale.
- [`image.show_image_list`](./docs/markdown_api/utils.image.md#function-show_image_list): Shows a grid of images, where each image is a Numpy array. The images can be either
- [`visualize.plot_features`](./docs/markdown_api/visualize.md#function-plot_features): Plot embeddings. This is a wrapper around : func : ` tsne. TSNE ` to make it easier to visualize the model's performance.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
