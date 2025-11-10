import torch
from collections import defaultdict
import copy
import torch.optim._functional as F

class SAM:
    def __init__(self, optimizer, model, rho=0.05, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.saved_state = defaultdict(dict)
        self.T = 25


    @torch.no_grad()
    def perturb_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]["eps"] = torch.zeros_like(p)
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
            eps[...] = p.grad[...]
            # if 'heads' not in n:
            #     eps.mul_(self.rho / grad_norm)
            # else:
            #     eps.mul_(0.2 / grad_norm)
            eps.mul_(self.rho / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.save_state()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def unperturb_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # print(n)
            p.sub_(self.state[p]["eps"])

    @torch.no_grad()
    def step(self):
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        loss = None
        closure = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            saved_state_with_grad = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    saved_state_with_grad.append(self.saved_state[p]["eps"])
                    d_p_list.append(p.grad)

                    state = self.optimizer.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # F.sgd(params_with_grad,
            #       d_p_list,
            #       momentum_buffer_list,
            #       weight_decay,
            #       momentum,
            #       lr,
            #       dampening,
            #       nesterov)

            for i, param in enumerate(params_with_grad):

                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                eps = saved_state_with_grad[i]
                with torch.no_grad():
                    sim = self.cal_sim(d_p.cpu().detach(), eps.cpu().detach())
                    # scale = 1
                    scale = min(1, self.T*(1-torch.exp(sim-1).cpu().detach().item()))
                    d_p = d_p * scale
                param.add_(d_p, alpha=-lr)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer.state[p]
                state['momentum_buffer'] = momentum_buffer
        self.optimizer.zero_grad()

    # @torch.no_grad()
    # def step(self):
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()


    @torch.no_grad()
    def save_state(self):
        self.saved_state = self.state.copy()
        # self.saved_state.popitem()

    @torch.no_grad()
    def cal_sim(self, a, b):
        a *= 100
        b *= 100
        a = a.view(1, -1)
        b = b.view(1, -1)
        sim = torch.nn.functional.cosine_similarity(a, b, dim=1)
        return sim
