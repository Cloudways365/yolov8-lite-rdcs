import torch
import ultralytics.nn.tasks as tasks

from .lite import LiteC2f, LiteConv

tasks.Conv = LiteConv
tasks.C2f = LiteC2f

_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

