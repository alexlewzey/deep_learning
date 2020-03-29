import torch


def show_methods(t: torch.Tensor):
    print(t)
    print(f'requires_grad: {t.requires_grad}')
    print(f'is_leaf: {t.is_leaf}')
    print(f'grad: {t.grad}')
    print(f'grad_fn: {t.grad_fn}')
    print()


x = torch.randn(10, requires_grad=False)  # if requires grad is true all subsequent operations have requires_grad=True

w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)

z_net = x * w + b
a_out = torch.sigmoid(z_net)

a_actual = torch.sigmoid(x * 2 + 3)

loss = torch.sum((a_out - a_actual) ** 2)

loss.backward()

vals = [x, w, b, z_net, a_out, loss]

for v in vals:
    show_methods(v)
