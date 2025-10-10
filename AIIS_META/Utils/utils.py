import numpy as np
import scipy
import scipy.signal
import json
import torch
from typing import Dict, List, Tuple, Optional
device = torch.device("cpu")

def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        import numpy as np
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return torch.as_tensor(x, device=device)


def discount_cumsum(rewards, discount):
    if type(rewards[0]) != torch.tensor:
        rewards[0] = to_tensor(rewards[0])

    returns = [rewards[-1]]
    for i in range(1, len(rewards)):
        if type(rewards[-i]) != torch.tensor:
            rewards[-i-1] = to_tensor(rewards[-i-1])
        returns.append(returns[-1]*discount + rewards[-i-1])
    
    returns.reverse()
    return returns

def module_device_dtype(module):
    # 1) 파라미터에서 추론
    it = module.parameters()
    first = next(it, None)
    if first is not None:
        return first.device, first.dtype
    # 2) 버퍼에서 추론(예: running_mean 등)
    itb = module.buffers()
    firstb = next(itb, None)
    if firstb is not None:
        return firstb.device, firstb.dtype
    # 3) 아무 것도 없으면 기본값
    return torch.device("cpu"), torch.get_default_dtype()

def stack_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = [[x[k] for x in tensor_dict_list]]
        ret[k] = v[0]
    return ret

def normalize_advantages(advantages):
    """
    Args:
        advantages (list[float]): list of advantages

    Returns:
        list[float]: normalized advantages
    """

    n = len(advantages[0])*len(advantages)
    if n == 0:
        return []

    mean_val = sum(sum(advantage) / n for advantage in advantages)
    var = sum(sum((x - mean_val) ** 2 for x in advantage) for advantage in advantages) / n
    std = var ** 0.5

    return [[(x - mean_val) / (std + 1e-8) for x in advantage] for advantage in advantages]


def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8

def to_numpy(x, *, dtype=np.float32):
    """Tensor/리스트/튜플/스칼라를 안전하게 numpy로 변환."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    if isinstance(x, (list, tuple)):
        # 내부에 텐서/스칼라 섞여 있어도 ok
        return np.asarray([to_numpy(v, dtype=dtype) for v in x], dtype=dtype)
    if isinstance(x, (int, float, np.number)):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if x is None:
        return None
    # dict나 기타 타입은 호출부에서 따로 처리
    raise TypeError(f"to_numpy: unsupported type {type(x)}")

def dict_to_numpy(d, *, dtype=np.float32):
    """dict의 값들을 numpy로 변환(값이 텐서/스칼라인 경우). 중첩 dict도 처리."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_numpy(v, dtype=dtype)
        elif isinstance(v, (torch.Tensor, np.ndarray, list, tuple, int, float, np.number)):
            out[k] = to_numpy(v, dtype=dtype)
        else:
            # 문자열 등은 그대로 둠
            out[k] = v
    return out
'''
def get_original_tf_name(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name given to the variable when creating it (i.e. name of the variable w/o the scope and the colons)
    """
    return name.split("/")[-1].split(":")[0]


def remove_scope_from_name(name, scope):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): full name of the variable with the scope removed
    """
    result = name.split(scope)[1]
    result = result[1:] if result[0] == '/' else result
    return result.split(":")[0]

def remove_first_scope_from_name(name):
    return name.replace(name + '/', "").split(":")[0]

def get_last_scope(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name of the last scope
    """
    return name.split("/")[-2]


def extract(x, *keys):
    """
    Args:
        x (dict or list): dict or list of dicts

    Returns:
        (tuple): tuple with the elements of the dict or the dicts of the list
    """
    if isinstance(x, dict):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def normalize_advantages(advantages):
    """
    Args:
        advantages (np.ndarray): np array with the advantages

    Returns:
        (np.ndarray): np array with the advantages normalized
    """
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8




def explained_variance_1d(ypred, y):
    """
    Args:
        ypred (np.ndarray): predicted values of the variable of interest
        y (np.ndarray): real values of the variable

    Returns:
        (float): variance explained by your estimator

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def concat_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of lists of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.concatenate([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU/CUDA).

    Args:
        seed (int): random seed
        deterministic (bool): make CUDA/cuDNN behavior deterministic (slower)

    Returns:
        None
    """
    import os, random
    import numpy as np
    import torch

    # 정규화(원하는 값 그대로 써도 되지만, 범위를 32-bit로 맞춰 저장성 보장)
    seed = int(seed) & 0xFFFFFFFF

    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU/CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Using seed {seed})")
'''