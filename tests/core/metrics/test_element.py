import copy
from functools import partial
from multiprocessing import Pool, set_start_method
from typing import Any, Dict, Type

import numpy as np
import pytest

from fastNLP import DataSet, Metric
from fastNLP.core.metrics.element import Element
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from tests.helpers.utils import skip_no_cuda
from .utils import find_free_network_port, setup_ddp, DemoMetric, DemoMetric2

if _NEED_IMPORT_TORCH:
    import torch

set_start_method('spawn', force=True)

NUM_PROCESSES = 2
pool = None


def _test(
    local_rank: int,
    world_size: int,
    device: 'torch.device',
    dataset: DataSet,
    metric_class: Type[Metric],
    metric_kwargs: Dict[str, Any],
) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        a = torch.tensor(dataset[i]['a'])
        metric.update(a.to(device=device))
    my_result = metric.get_metric()
    if local_rank == 0:
        np.testing.assert_almost_equal(my_result['a'], [10, 14, 18, 22])


@pytest.fixture(scope='class', autouse=True)
def pre_process():
    global pool
    pool = Pool(processes=NUM_PROCESSES)
    master_port = find_free_network_port()
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, master_port)
                             for rank in range(NUM_PROCESSES)])
    yield
    pool.close()
    pool.join()


@pytest.mark.torch
def test_list_ddp() -> None:
    # 该测试的目的
    skip_no_cuda()
    global pool
    metric_kwargs = {}
    metric_kwargs['aggregate_when_get_metric'] = True
    metric_kwargs['backend'] = 'torch'
    processes = NUM_PROCESSES
    dataset = DataSet(
        {'a': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]})
    pool.starmap(
        partial(
            _test,
            dataset=dataset,
            metric_class=DemoMetric2,
            metric_kwargs=metric_kwargs,
        ), [(rank, processes, torch.device(f'cuda:{rank}'))
            for rank in range(processes)])


@pytest.mark.torch
def test_torch_return_element():
    # 测试的目的是为了方式直接返回了Element对象
    metric = DemoMetric('torch')
    metric.update(a=1)
    res = metric.get_metric()
    metric.reset()
    assert res['a'] == 1


@pytest.mark.torch
def test_getter_setter():
    ele = Element(
        name='a',
        value=[0, 1, 4, 2, 3],
        aggregate_method='sum',
        backend='torch')
    x = ele[2]
    assert x == 4
    ele[4] = 100
    assert ele.to_list() == [0, 1, 4, 2, 100]
    ele2 = Element(name='b', value=0, aggregate_method='sum', backend='torch')
    with pytest.raises(TypeError):
        ele2[0]
    with pytest.raises(TypeError):
        ele2[0] = 3
