import os
import socket
import sys
from typing import Union

import numpy as np

from fastNLP.core.metrics import Metric
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch


def setup_ddp(rank: int, world_size: int, master_port: int) -> None:
    """Setup ddp environment."""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    if torch.distributed.is_available() and sys.platform not in ('win32',
                                                                 'cygwin'):
        torch.distributed.init_process_group(
            'gloo', rank=rank, world_size=world_size)


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a
    real master node but have to set the `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def _assert_allclose(my_result: Union[float, np.ndarray],
                     sklearn_result: Union[float, np.ndarray],
                     atol: float = 1e-8) -> None:
    """测试对比结果，这里不用非得是必须数组且维度对应，一些其他情况例如 np.allclose(np.array([[1e10, ], ]),
    1e10+1) 也是 True.

    :param my_result: 可以不限设备等
    :param sklearn_result:
    :param atol:
    :return:
    """
    assert np.allclose(a=my_result, b=sklearn_result, atol=atol)


class DemoMetric(Metric):

    def __init__(self, backend):
        super(DemoMetric, self).__init__(backend=backend)
        self.register_element('a', 0, aggregate_method='sum')

    def update(self, a):
        self.a += a

    def get_metric(self) -> dict:
        return {'a': self.a}


class DemoMetric2(Metric):

    def __init__(self, backend, aggregate_when_get_metric):
        super(DemoMetric2, self).__init__(
            backend=backend,
            aggregate_when_get_metric=aggregate_when_get_metric)
        self.register_element('a', [0, 0, 0, 0], aggregate_method='sum')

    def update(self, a):
        self.a += a

    def get_metric(self) -> dict:
        return {'a': self.a}
