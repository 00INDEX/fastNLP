"""这个文件测试多卡情况下使用 deepspeed 的情况::

    >>> # 测试直接使用多卡
    >>> python _test_trainer_deepspeed.py
    >>> # 测试通过 deepspeed 拉起
    >>> deepspeed _test_trainer_deepspeed.py
"""
import os
import sys
import argparse
from dataclasses import dataclass

path = os.path.abspath(__file__)
folders = path.split(os.sep)
for folder in list(folders[::-1]):
    if 'fastnlp' not in folder.lower():
        folders.pop(-1)
    else:
        break
path = os.sep.join(folders)
sys.path.extend([path, os.path.join(path, 'fastNLP')])

from torch.optim import Adam
from torch.utils.data import DataLoader

from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1


@dataclass
class TrainDeepSpeedConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2


def test_trainer_deepspeed(
    device,
    callbacks,
    strategy,
    config,
    n_epochs=2,
):
    model = TorchNormalModel_Classification_1(
        num_labels=TrainDeepSpeedConfig.num_labels,
        feature_dimension=TrainDeepSpeedConfig.feature_dimension)
    optimizers = Adam(params=model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 20),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 12),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True)
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainDeepSpeedConfig.evaluate_every
    metrics = {'acc': Accuracy()}
    if config is not None:
        config[
            'train_micro_batch_size_per_gpu'] = TrainDeepSpeedConfig.batch_size
    trainer = Trainer(
        model=model,
        driver='deepspeed',
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=evaluate_dataloaders,
        evaluate_every=evaluate_every,
        metrics=metrics,
        output_mapping={'preds': 'pred'},
        n_epochs=n_epochs,
        callbacks=callbacks,
        deepspeed_kwargs={
            'strategy': strategy,
            'config': config
        })
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input trainer parameters.')
    parser.add_argument('-d', '--device', type=int, nargs='+', default=None)

    args = parser.parse_args()
    if args.device is None:
        args.device = [4, 5]
    # device = [0,1,3]
    callbacks = [
        # RecordMetricCallback(monitor="acc#acc", metric_threshold=0.0, larger_better=True),
        RichCallback(5),
    ]
    config = None
    test_trainer_deepspeed(
        device=args.device,
        callbacks=callbacks,
        strategy='deepspeed',
        config=config,
        n_epochs=5,
    )
