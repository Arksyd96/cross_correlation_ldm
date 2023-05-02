import pytorch_lightning as pl

class EMA(pl.Callback):
    def __init__(self, decay_rate=0.999):
        super().__init__()
        self.decay_rate = decay_rate

    def on_train_start(self, trainer, pl_module):
        self.shadow_params = {}
        for name, param in pl_module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                decay = self.decay_rate
                self.shadow_params[name] -= (1 - decay) * (self.shadow_params[name] - param.data)

    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.grad = self.shadow_params[name] - param.data

    def on_train_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            param.data.copy_(self.shadow_params[name])