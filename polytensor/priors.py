def uniformBernoulli(n, device, bias=0.5, **kwargs):
    """Uniform prior on the interval [0,1]."""
    if basis == Basis.spin:
        return torch.bernoulli(torch.ones(n, device=device) * bias) * 2 - 1
    return torch.bernoulli(torch.ones(n, device=device) * bias)


class Uniform:
    def __init__(self, n, device, basis=Basis.standard, bias=0.5, **kwargs):
        self.n = n
        self.device = device
        self.basis = basis
        self.bias = bias

    def __call__(self, x=None, **kwargs):
        return uniform(self.n, self.device, self.basis, self.bias)
