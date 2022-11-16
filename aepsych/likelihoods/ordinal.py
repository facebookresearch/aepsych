from typing import Callable, Optional

import gpytorch
import torch
from gpytorch.likelihoods import Likelihood
from torch.distributions import Categorical, Normal


class OrdinalLikelihood(Likelihood):
    """
    Ordinal likelihood, suitable for rating models (e.g. likert scales). Formally,
    .. math:: z_k(x\\mid f) := p(d_k < f(x) \\le d_{k+1}) = \\sigma(d_{k+1}-f(x)) - \\sigma(d_{k}-f(x)),
    where :math:`\\sigma()` is the link function (equivalent to the perceptual noise
    distribution in psychophysics terms), :math:`f(x)` is the latent GP evaluated at x,
    and :math:`d_k` is a learned cutpoint parameter for each level.
    """

    def __init__(self, n_levels: int, link: Optional[Callable] = None):
        super().__init__()
        self.n_levels = n_levels
        self.register_parameter(
            name="raw_cutpoint_deltas",
            parameter=torch.nn.Parameter(torch.abs(torch.randn(n_levels - 2))),
        )
        self.register_constraint("raw_cutpoint_deltas", gpytorch.constraints.Positive())
        self.link = link or Normal(0, 1).cdf

    @property
    def cutpoints(self):
        cutpoint_deltas = self.raw_cutpoint_deltas_constraint.transform(
            self.raw_cutpoint_deltas
        )
        # for identification, the first cutpoint is 0
        return torch.cat((torch.tensor([0]), torch.cumsum(cutpoint_deltas, 0)))

    def forward(self, function_samples, *params, **kwargs):

        # this whole thing can probably be some clever batched thing, meh
        probs = torch.zeros(*function_samples.size(), self.n_levels)

        probs[..., 0] = self.link(self.cutpoints[0] - function_samples)

        for i in range(1, self.n_levels - 1):
            probs[..., i] = self.link(self.cutpoints[i] - function_samples) - self.link(
                self.cutpoints[i - 1] - function_samples
            )
        probs[..., -1] = 1 - self.link(self.cutpoints[-1] - function_samples)
        res = Categorical(probs=probs)
        return res

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        n_levels = config.getint(classname, "n_levels")
        link = config.getobj(classname, "link", fallback=None)
        return cls(n_levels=n_levels, link=link)
