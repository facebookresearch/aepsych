import numpy as np
from aepsych.utils import _dim_grid, get_lse_contour
from scipy.stats import bernoulli, norm, pearsonr


class Problem:
    def __init__(self, lb, ub, **options):
        assert len(lb) == len(ub), "bounds should be same size"
        dim = len(lb)
        self.options = options
        gridsize = self.options.get("gridsize", 10)
        self.eval_grid = _dim_grid(
            lower=lb, upper=ub, dim=dim, gridsize=gridsize
        ).squeeze()

    def f(self, x):
        raise NotImplementedError

    def p(self, x):
        return norm.cdf(self.f(x))

    def sample_y(self, x):
        return bernoulli.rvs(self.p(x))

    def f_hat(self, strat):
        f_hat, _ = strat.predict(self.eval_grid)
        return f_hat

    def f_true(self):
        return self.f(self.eval_grid)

    def evaluate(self, strat):
        # always eval f
        f_true = self.f_true().numpy()
        f_hat = self.f_hat(strat).detach().numpy()
        assert (
            f_true.shape == f_hat.shape
        ), f"f_true.shape=={f_true.shape} != f_hat.shape=={f_hat.shape}"
        p_true = norm.cdf(f_true)
        p_hat = norm.cdf(f_hat)
        mae_f = np.mean(np.abs(f_true - f_hat))
        mse_f = np.mean((f_true - f_hat) ** 2)
        max_abs_err_f = np.max(np.abs(f_true - f_hat))
        corr_f = pearsonr(f_true.flatten(), f_hat.flatten())[0]
        mae_p = np.mean(np.abs(p_true - p_hat))
        mse_p = np.mean((p_true - p_hat) ** 2)
        max_abs_err_p = np.max(np.abs(p_true - p_hat))
        corr_p = pearsonr(p_true.flatten(), p_hat.flatten())[0]

        # eval in samp-based expectation over posterior instead of just mean
        fsamps = strat.sample(self.eval_grid, num_samples=1000).detach().numpy()
        ferrs = fsamps - f_true[None, :]
        miae_f = np.mean(np.abs(ferrs))
        mise_f = np.mean(ferrs ** 2)

        perrs = norm.cdf(fsamps) - norm.cdf(f_true[None, :])
        miae_p = np.mean(np.abs(perrs))
        mise_p = np.mean(perrs ** 2)

        metrics = {
            "mean_abs_err_f": mae_f,
            "mean_integrated_abs_err_f": miae_f,
            "mean_square_err_f": mse_f,
            "mean_integrated_square_err_f": mise_f,
            "max_abs_err_f": max_abs_err_f,
            "pearson_corr_f": corr_f,
            "mean_abs_err_p": mae_p,
            "mean_integrated_abs_err_p": miae_p,
            "mean_square_err_p": mse_p,
            "mean_integrated_square_err_p": mise_p,
            "max_abs_err_p": max_abs_err_p,
            "pearson_corr_p": corr_p,
        }

        return metrics


class LSEProblem:
    def evaluate(self, strat):
        metrics = super().evaluate(strat)

        thresh = self.options.get("thresh", 0.75)
        gridsize = self.options.get("gridsize", 10)
        post_mean, _ = strat.predict(self.eval_grid)

        dim = self.eval_grid.shape[1]
        post_mean_reshape = post_mean.reshape((gridsize,) * dim)
        phi_post_mean = norm.cdf(post_mean_reshape.detach().numpy())
        # assume mono_dim is last dim (TODO make this better)

        x1 = _dim_grid(
            lower=strat.lb.numpy()[-1],
            upper=strat.ub.numpy()[-1],
            dim=1,
            gridsize=gridsize,
        ).squeeze()
        x2_hat = get_lse_contour(phi_post_mean, x1, level=thresh, lb=-1.0, ub=1.0)

        true_f = self.f(self.eval_grid)

        true_f_reshape = true_f.reshape((gridsize,) * dim)

        true_x2 = get_lse_contour(
            norm.cdf(true_f_reshape), x1, level=thresh, lb=-1.0, ub=1.0
        )
        assert x2_hat.shape == true_x2.shape, (
            "x2_hat.shape != true_x2.shape, something went wrong!"
            + f"x2_hat.shape={x2_hat.shape}, true_x2.shape={true_x2.shape}"
        )
        mae = np.mean(np.abs(true_x2 - x2_hat))
        mse = np.mean((true_x2 - x2_hat) ** 2)
        max_abs_err = np.max(np.abs(true_x2 - x2_hat))

        metrics["mean_abs_err_thresh"] = mae
        metrics["mean_square_err_thresh"] = mse
        metrics["max_abs_err_thresh"] = max_abs_err

        if dim != 1:
            corr = pearsonr(true_x2.flatten(), x2_hat.flatten())[0]
            metrics["pearson_corr_thresh"] = corr

        # now construct integrated error on thresh
        fsamps = strat.sample(self.eval_grid, num_samples=1000).detach().numpy()

        square_samps = [s.reshape((gridsize,) * strat.modelbridge.dim) for s in fsamps]
        contours = np.stack(
            [
                get_lse_contour(norm.cdf(s), x1, level=thresh, mono_dim=-1, lb=-1, ub=1)
                for s in square_samps
            ]
        )

        thresh_err = contours - true_x2[None, :]

        metrics["mean_integrated_abs_err_thresh"] = np.mean(np.abs(thresh_err))
        metrics["mean_integrated_square_err_thresh"] = np.mean(thresh_err ** 2)

        return metrics
