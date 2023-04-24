import click
from data.data_factory import data_factory
from data.transforms import transform_factory
from data.collate import collate_factory
from models.model_factory import model_factory
from pipeline.lightning import LightningPipeline

"""
python run_model.py 
--root-dir=/home/uj43ugat/icdar/data/ICDAR2017_CLaMM_Training
--label-path=/home/uj43ugat/icdar/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv
--max-epochs=10
--model-name=byol
"""
@click.command()
@click.option('--root-dir', help='Dataset root directory', required=True)
@click.option('--label-path', help='Label CSV filepath', required=True)
@click.option('--dataset', default='icdar', help='Dataset name')
@click.option('--model-name', default='simclr', help='Model to run')
@click.option('--mode', default='train', help='Execution mode (train, test)')
@click.option('--max-epochs', default=100, type=int, help='Maximum number of epochs')
@click.option('--batch-size', default=32, type=int, help='Batch size')
def execute(root_dir, label_path, dataset, model_name, mode, max_epochs, batch_size):

    transforms = transform_factory(model_name, mode)
    
    collate_fn = collate_factory(model_name)
    
    data_loader = data_factory(dataset, root_dir, label_path, transforms, mode, batch_size, collate_fn)
    
    model_class = model_factory(
        model_name,
        gpus=-1,
        num_samples=len(data_loader.get(mode).dataset),
        max_epochs=max_epochs,
        batch_size=batch_size,
        dataset=dataset
    )
    
    print(model_class)
    print(transforms)
    print(data_loader)

    pipeline = LightningPipeline(
        root_dir,
        model_class,
        mode,
        data_loader,
        max_epochs,
        batch_size
    )

    pipeline.run()


if __name__ == '__main__':
    execute()
