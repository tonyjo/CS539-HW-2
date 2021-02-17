import torch


def _totensor(x, dtype=torch.float32):
    if not torch.is_tensor(x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=dtype)
        else:
            x = torch.tensor([x], dtype=dtype)
    return x

def _squareroot(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)

def _hashmap(x):
    # List (key, value, key, value, ....)
    return_x = {}
    if len(x)%2 == 0:
        for i in range(0, len(x)-1, 2):
            return_x[x[i]] = torch.tensor(x[i+1])
    else:
        raise IndexError('Un-even key-value pairs')

    return return_x
