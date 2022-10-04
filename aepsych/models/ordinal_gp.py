import gpytorch
import torch
from aepsych.likelihoods import OrdinalLikelihood
from aepsych.models import GPClassificationModel


class OrdinalGPModel(GPClassificationModel):
    """
    Convenience wrapper for GPClassificationModel that hardcodes
    an ordinal likelihood, better priors for this setting, and
    adds a convenience method for computing outcome probabilities.

    TODO: at some point we should refactor posteriors so that things like
    OrdinalPosterior and MonotonicPosterior don't have to have their own
    model classes.
    """

    outcome_type = "ordinal"

    def __init__(self, likelihood=None, *args, **kwargs):
        covar_module = kwargs.pop("covar_module", None)
        dim = kwargs.get("dim")
        if covar_module is None:

            ls_prior = gpytorch.priors.GammaPrior(concentration=1.5, rate=3.0)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            ls_constraint = gpytorch.constraints.Positive(
                transform=None, initial_value=ls_prior_mode
            )

            # no outputscale due to shift identifiability in d.
            covar_module = gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim,
            )

        if likelihood is None:
            likelihood = OrdinalLikelihood(n_levels=5)
        super().__init__(
            *args,
            covar_module=covar_module,
            likelihood=likelihood,
            **kwargs,
        )

    def predict_probs(self, xgrid):
        fmean, fvar = self.predict(xgrid)
        fsd = torch.sqrt(1 + fvar)
        probs = torch.zeros(*fmean.size(), self.likelihood.n_levels)

        probs[..., 0] = self.likelihood.link(
            (self.likelihood.cutpoints[0] - fmean) / fsd
        )

        for i in range(1, self.likelihood.n_levels - 1):
            probs[..., i] = self.likelihood.link(
                (self.likelihood.cutpoints[i] - fmean) / fsd
            ) - self.likelihood.link((self.likelihood.cutpoints[i - 1] - fmean) / fsd)
        probs[..., -1] = 1 - self.likelihood.link(
            (self.likelihood.cutpoints[-1] - fmean) / fsd
        )
        return probs
