import os
import click
import joblib
import optuna
import numpy as np
from functools import partial
from datetime import datetime
import torchvision.transforms as T
from data.data_factory import data_factory
from data.transforms import transform_factory
from data.collate import collate_factory
from models.model_factory import model_factory
from pipeline.lightning import LightningPipeline
from data.augment import Dilation, GaussianNoise

# Values are same because we have grayscale images
icdar_mean = [0.7013, 0.7013, 0.7013]
icdar_std = [0.2510, 0.2510, 0.2510]


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
    
    now = datetime.now()
    study = optuna.create_study(
        direction='minimize',
        # storage=f'sqlite:///optuna-iwfa028h.db',
        study_name=f'{model_name}_{dataset}_{now.strftime("%Y%m%d%H%M%S")}'
    )
    study.optimize(partial_objective, n_trials=20)
    
    joblib.dump(study, study_file_path)


def objective(root_dir, label_path, dataset, model_name, mode, num_cpus, trial):
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    max_epochs = trial.suggest_int('max_epochs', 50, 100)
    temperature = trial.suggest_float('temperature', 0.3, 0.8)
    dilation_size = trial.suggest_int('dilation_size', 1, 5, step=2)
    blur_kernel_size = trial.suggest_int('blur_kernel_size', 5, 50, step=2)
    rotation_angle = trial.suggest_int('rotation_angle', 0, 30)
    normalize_bool = trial.suggest_categorical('normalize_bool', [True, False])
    
    simclr_additional = T.Compose([
        T.RandomApply([T.GaussianBlur(kernel_size=(blur_kernel_size, blur_kernel_size))], p=0.5),
        T.RandomApply([T.RandomRotation(degrees=rotation_angle)], p=0.5),
        T.RandomErasing(p=0.5),
        T.RandomApply([Dilation(dilation_size)], p=0.5),
        T.RandomApply([T.Normalize(icdar_mean, icdar_std)], p=int(normalize_bool))
    ])

    transforms = transform_factory(model_name, mode, simclr_additional)
    
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