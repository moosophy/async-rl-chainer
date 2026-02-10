import numpy as np
from chainer import cuda
from chainer import optimizer

class RMSpropAsync(optimizer.GradientMethod):
    """RMSprop for asynchronous A3C with shared-memory support."""

    def __init__(self, lr=7e-4, alpha=0.99, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self._states = {}  # <- needed by async code
        

    def setup(self, link):
        super().setup(link)
        for idx, param in enumerate(link.params()):
            param.update_rule = self.create_update_rule()
            param.update_rule.init_state(param)
            self._states[idx] = param.update_rule.state
        return self

    def create_update_rule(self):
        class Rule:
            def __init__(self, lr, alpha, eps):
                self.lr = lr
                self.alpha = alpha
                self.eps = eps
                self.state = None

            def init_state(self, param):
                xp = cuda.get_array_module(param.data)
                self.state = {'ms': xp.zeros_like(param.data)}

            def update_one_cpu(self, param):
                ms = self.state['ms']
                grad = param.grad
                ms *= self.alpha
                ms += (1 - self.alpha) * grad * grad
                param.data -= self.lr * grad / np.sqrt(ms + self.eps)

            def update_one_gpu(self, param):
                ms = self.state['ms']
                grad = param.grad
                xp = cuda.get_array_module(param.data)
                ms *= self.alpha
                ms += (1 - self.alpha) * grad * grad
                param.data -= self.lr * grad / xp.sqrt(ms + self.eps)

        return Rule(self.lr, self.alpha, self.eps)
