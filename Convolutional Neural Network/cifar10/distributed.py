import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module

"""
Distributed Traning enhancement
"""

class DistributedDataParallel(Module):
    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        assert dist._backend == dist.dist_backend.GLOO, "Only Support Gloo backend" 
        self.module = module

        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            # Sync params
            dist.broadcast(p, 0)

        def allreduce_params():
            if (self.needs_reduction):
                self.needs_reduction = False

                buckets = {}

                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)

                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)
