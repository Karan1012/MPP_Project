import math

import torch
import numpy as np
import torch.multiprocessing as mp

class SharedASGD(torch.optim.Optimizer):
    """Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Acceleration of stochastic approximation by averaging:
        http://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(SharedASGD, self).__init__(params, defaults)


        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['eta'] = mp.Value('d', group['lr'])
                state['step'] = torch.zeros(1)
                state['mu'] = mp.Value('d', 1)
                state['ax'] = p.data.new().resize_as_(p.data).zero_()


    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
            #    state['eta'].share_memory_()
                state['step'].share_memory_()
                state['ax'].share_memory_()
             #   state['mu'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[p]


                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decay term
                p.data.mul_(1 - group['lambd'] * state['eta'].value)

                # update parameter
                p.data.add_(-state['eta'].value, grad)

                # averaging
                if state['mu'].value != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu'].value))
                else:
                    state['ax'].copy_(p.data)

                # update eta and mu
                step = np.array(state['step'])[0]
                state['eta'].value = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * step), group['alpha']))
                state['mu'].value = 1 / max(1, step - group['t0'])

        return loss
