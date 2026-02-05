import numpy as np
import scipy
import scipy.signal
import torch
device = torch.device("cpu")

def to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        import numpy as np
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return torch.as_tensor(x, device=device)

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


def discount_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def module_device_dtype(module):
    # 1) Infer from parameters
    it = module.parameters()
    first = next(it, None)
    if first is not None:
        return first.device, first.dtype
    # 2) Infer from buffers (e.g., running_mean)
    itb = module.buffers()
    firstb = next(itb, None)
    if firstb is not None:
        return firstb.device, firstb.dtype
    # 3) Fallback to default
    return torch.device("cpu"), torch.get_default_dtype()

def concat_envs(env_dict_lst):
    """
    Args:
        env_dict_lst (list) : list of dicts of lists of envs

    Returns:
        (dict) : dict of lists of envs
    """
    keys = list(env_dict_lst[0].keys())
    ret = dict()
    for k in keys:
        example = env_dict_lst[0][k]
        if isinstance(example, dict):
            v = concat_envs([x[k] for x in env_dict_lst])
        else:
            v = np.concatenate([x[k] for x in env_dict_lst])
        ret[k] = v
    return ret
def concat_agernt_infos(agent_info_list):
    """
    Args:
        agent_info_list (list) : list of dicts of lists of agent_infos

    Returns:
        (dict) : dict of lists of agent_infos
    """
    keys = list(agent_info_list[0].keys())
    ret = dict()
    temp = []
    for k in keys:
        for x in agent_info_list:
            temp.extend(x[k])
        ret[k] = temp
    return ret

def stack_envs(envs_dict_list):
    """
    Args:
        envs_dict_list (list) : list of dicts of envs

    Returns:
        (dict) : dict of lists of envs
    """
    keys = list(envs_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = envs_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_envs([x[k] for x in envs_dict_list])
        else:
            v = np.asarray([x[k] for x in envs_dict_list])
        ret[k] = v
    return ret

def stack_agent_infos(agent_info_dict_list):
    """
    Args:
        agent_info_dict_list (list) : list of dicts of agent_infos

    Returns:
        (dict) : dict of lists of agent_infos
    """
    keys = list(agent_info_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = agent_info_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_agent_infos([x[k] for x in agent_info_dict_list])
        else:
            v = [x[k] for x in agent_info_dict_list]
        ret[k] = v
    return ret


def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8

def to_numpy(x, *, dtype=np.float32):
    """Safely convert tensor/list/tuple/scalar to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    if isinstance(x, (list, tuple)):
        # Nested tensors/scalars are OK
        return np.asarray([to_numpy(v, dtype=dtype) for v in x], dtype=dtype)
    if isinstance(x, (int, float, np.number)):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if x is None:
        return None
    # dict or other types should be handled by the caller
    raise TypeError(f"to_numpy: unsupported type {type(x)}")

def dict_to_numpy(d, *, dtype=np.float32):
    """Convert dict values to numpy when values are tensors/scalars. Handles nested dicts."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_numpy(v, dtype=dtype)
        elif isinstance(v, (torch.Tensor, np.ndarray, list, tuple, int, float, np.number)):
            out[k] = to_numpy(v, dtype=dtype)
        else:
            # Keep strings and other types as-is
            out[k] = v
    return out
