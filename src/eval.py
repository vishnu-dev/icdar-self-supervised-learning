import click
from data.data_factory import data_factory
from data.transforms import transform_factory
from data.collate import collate_factory
from pipeline.lightning import LightningPipeline
from models.backbone_factory import backbone_factory
from models.model_factory import model_factory

"""
python eval.py --root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training --label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv --max-epochs=100 --batch-size=32 --model-name=mae
"""


@click.command()
@click.option("--root-dir", help="Dataset root directory", required=True)
@click.option("--label-path", help="Label CSV filepath", required=True)
@click.option("--dataset", default="icdar", help="Dataset name")
@click.option("--model-name", default="downstream_linear", help="Model to evaluate")
@click.option("--base-model-name", default="simclr", help="Model to use for getting embeddings")
@click.option("--base-checkpoint", default="", help="Pretext model checkpoint")
@click.option("--mode", default="train", help="Execution mode (train, test)")
@click.option("--max-epochs", default=100, type=int, help="Maximum number of epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--num-cpus", default=8, type=int, help="Number of CPUs for data loading")
@click.option("--learning-rate", default=0.001, type=float, help="Learning rate")
def execute(
    root_dir,
    label_path,
    dataset,
    model_name,
    base_model_name,
    backbone_checkpoint,
    mode,
    max_epochs,
    batch_size,
    num_cpus,
    learning_rate,
):
    transforms = transform_factory(model_name, mode)

    collate_fn = collate_factory(model_name)

    data_loader = data_factory(
        dataset,
        root_dir,
        label_path,
        transforms,
        mode,
        batch_size,
        collate_fn,
        num_cpus,
    )

    base_model = backbone_factory(
        base_model_name,
        backbone_checkpoint,
        gpus=-1,
        num_samples=len(data_loader.get(mode).dataset),
        max_epochs=max_epochs,
        batch_size=batch_size,
        dataset=dataset,
    )

    model_class = model_factory(
        model_name,
        batch_size=batch_size,
        base_model=base_model
    )

    pipeline = LightningPipeline(
        root_dir, model_class, mode, data_loader, max_epochs, batch_size
    )

    pipeline.run()


if __name__ == "__main__":
    execute()
