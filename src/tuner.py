import os
import click
import joblib
import optuna
from data.data_factory import data_factory
from data.transforms import transform_factory
from data.collate import collate_factory
from models.model_factory import model_factory
from pipeline.lightning import LightningPipeline
from functools import partial


@click.command()
@click.option('--root-dir', help='Dataset root directory', required=True)
@click.option('--label-path', help='Label CSV filepath', required=True)
@click.option('--dataset', default='icdar', help='Dataset name')
@click.option('--model-name', default='simclr', help='Model to run')
@click.option('--mode', default='train', help='Execution mode (train, test)')
@click.option('--num-cpus', default=8, type=int, help='Number of CPUs for data loading')
def execute(root_dir, label_path, dataset, model_name, mode, num_cpus):
    
    study_file_path = os.path.abspath(os.path.join(
        root_dir, '..', 'trained_models', f'{model_name}_{dataset}_study.pkl'
    ))
    print('Saving study at: ', study_file_path)
    
    partial_objective = partial(objective, root_dir, label_path, dataset, model_name, mode, num_cpus)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(partial_objective, n_trials=20)
    
    joblib.dump(study, study_file_path)


def objective(root_dir, label_path, dataset, model_name, mode, num_cpus, trial):
    
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1)
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    max_epochs = trial.suggest_int('max_epochs', 30, 100)
    temperature = trial.suggest_float('temperature', 0.3, 0.8)

    transforms = transform_factory(model_name, mode)
    
    collate_fn = collate_factory(model_name)
    
    data_loader = data_factory(dataset, root_dir, label_path, transforms, mode, batch_size, collate_fn, num_cpus)
    
    model_class = model_factory(
        model_name,
        gpus=-1,
        num_samples=len(data_loader.get(mode).dataset),
        max_epochs=max_epochs,
        batch_size=batch_size,
        dataset=dataset,
        learning_rate=learning_rate,
        temperature=temperature
    )

    pipeline = LightningPipeline(
        root_dir,
        model_class,
        mode,
        data_loader,
        max_epochs,
        batch_size
    )

    pipeline.run()
    
    return pipeline.trainer.callback_metrics["val_loss"].item()


if __name__ == '__main__':
    execute()