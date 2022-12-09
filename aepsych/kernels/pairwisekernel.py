import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import lazify


class PairwiseKernel(Kernel):
    """
    Wrapper to convert a kernel K on R^k to a kernel K' on R^{2k}, modeling
    functions of the form g(a, b) = f(a) - f(b), where f ~ GP(mu, K).

    Since g is a linear combination of Gaussians, it follows that g ~ GP(0, K')
    where K'((a,b), (c,d)) = K(a,c) - K(a, d) - K(b, c) + K(b, d).

    """

    def __init__(self, latent_kernel, is_partial_obs=False, **kwargs):
        super(PairwiseKernel, self).__init__(**kwargs)

        self.latent_kernel = latent_kernel
        self.is_partial_obs = is_partial_obs

    def forward(self, x1, x2, diag=False, **params):
        r"""
        TODO: make last_batch_dim work properly

        d must be 2*k for integer k, k is the dimension of the latent space
        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `diag`: `n` or `b x n`
        """
        if self.is_partial_obs:
            d = x1.shape[-1] - 1
            assert d == x2.shape[-1] - 1, "tensors not the same dimension"
            assert d % 2 == 0, "dimension must be even"

            k = int(d / 2)

            # special handling for kernels that (also) do funky
            # things with the input dimension
            deriv_idx_1 = x1[..., -1][:, None]
            deriv_idx_2 = x2[..., -1][:, None]

            a = torch.cat((x1[..., :k], deriv_idx_1), dim=1)
            b = torch.cat((x1[..., k:-1], deriv_idx_1), dim=1)
            c = torch.cat((x2[..., :k], deriv_idx_2), dim=1)
            d = torch.cat((x2[..., k:-1], deriv_idx_2), dim=1)

        else:
            d = x1.shape[-1]

            assert d == x2.shape[-1], "tensors not the same dimension"
            assert d % 2 == 0, "dimension must be even"

            k = int(d / 2)

            a = x1[..., :k]
            b = x1[..., k:]
            c = x2[..., :k]
            d = x2[..., k:]

        if not diag:
            return (
                lazify(self.latent_kernel(a, c, diag=diag, **params))
                + lazify(self.latent_kernel(b, d, diag=diag, **params))
                - lazify(self.latent_kernel(b, c, diag=diag, **params))
                - lazify(self.latent_kernel(a, d, diag=diag, **params))
            )
        else:
            return (
                self.latent_kernel(a, c, diag=diag, **params)
                + self.latent_kernel(b, d, diag=diag, **params)
                - self.latent_kernel(b, c, diag=diag, **params)
                - self.latent_kernel(a, d, diag=diag, **params)
            )
