"""One-device wrapper preserving DDP-compatible checkpoint key names."""
from torch import nn


class SingleDeviceModule(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module=module

    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)
