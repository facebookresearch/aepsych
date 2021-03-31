import matplotlib.pyplot as plt
import numpy as np
from aepsych.utils import get_lse_interval, _dim_grid, get_lse_contour
from scipy.stats import norm


def plot_strat_1d(
    strat,
    title,
    ax=None,
    true_testfun=None,
    cred_level=0.95,
    target_level=0.75,
    xlabel="Intensity (abstract)",
    gridsize=30,
):

    x, y = strat.x, strat.y

    if ax is None:
        fig, ax = plt.subplots()

    grid = _dim_grid(modelbridge=strat.modelbridge, gridsize=gridsize)
    samps = norm.cdf(strat.modelbridge.sample(grid, num_samples=10000))
    phimean = samps.mean(0)
    upper = np.quantile(samps, cred_level, axis=0)
    lower = np.quantile(samps, 1 - cred_level, axis=0)

    ax.plot(np.squeeze(grid), phimean)
    ax.fill_between(
        np.squeeze(grid),
        lower,
        upper,
        alpha=0.3,
        hatch="///",
        edgecolor="gray",
        label=f"{cred_level*100:.0f}% posterior mass",
    )
    if target_level is not None:
        from aepsych.utils import interpolate_monotonic

        threshold_samps = [
            interpolate_monotonic(
                grid.squeeze().numpy(), s, target_level, strat.lb[0], strat.ub[0]
            )
            for s in samps
        ]
        thresh_med = np.mean(threshold_samps)
        thresh_lower = np.quantile(threshold_samps, q=1 - cred_level)
        thresh_upper = np.quantile(threshold_samps, q=cred_level)

        ax.errorbar(
            thresh_med,
            target_level,
            xerr=np.r_[thresh_med - thresh_lower, thresh_upper - thresh_med][:, None],
            capsize=5,
            elinewidth=1,
            label=f"Est. {target_level*100:.0f}% threshold \n(with {cred_level*100:.0f}% posterior \nmass marked)",
        )

    if true_testfun is not None:
        # true_testfun = lambda x: 3*x
        true_f = norm.cdf(true_testfun(grid))
        ax.plot(grid, true_f.squeeze(), label="True function")
        if target_level is not None:
            true_thresh = interpolate_monotonic(
                grid.squeeze().numpy(),
                true_f.squeeze(),
                target_level,
                strat.lb[0],
                strat.ub[0],
            )

            ax.plot(
                true_thresh.item(),
                target_level,
                "o",
                label=f"True {target_level*100:.0f}% threshold",
            )

    ax.scatter(
        x[y == 0, 0],
        np.zeros_like(x[y == 0, 0]),
        marker=3,
        color="r",
        label="Nondetected trials",
    )
    ax.scatter(
        x[y == 1, 0],
        np.zeros_like(x[y == 1, 0]),
        marker=3,
        color="b",
        label="Detected trials",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Response Probability")

    ax.set_title(title)


def plot_strat_2d(
    strat,
    title,
    ax=None,
    true_testfun=None,
    cred_level=0.95,
    target_level=0.75,
    xlabel="Context (abstract)",
    ylabel="Intensity (abstract)",
    flipx=False,
    logx=False,
    gridsize=30,
):

    x, y = strat.x, strat.y

    if ax is None:
        fig, ax = plt.subplots()

    grid = _dim_grid(modelbridge=strat.modelbridge, gridsize=gridsize)
    fmean, fvar = strat.modelbridge.predict(grid)
    phimean = norm.cdf(fmean.reshape(gridsize, gridsize).detach().numpy()).T

    if flipx:
        extent = np.r_[strat.lb[0], strat.ub[0], strat.ub[1], strat.lb[1]]
        _ = ax.imshow(phimean, aspect="auto", origin="upper", extent=extent, alpha=0.5)
    else:
        extent = np.r_[strat.lb[0], strat.ub[0], strat.lb[1], strat.ub[1]]
        _ = ax.imshow(phimean, aspect="auto", origin="lower", extent=extent, alpha=0.5)

    # hacky relabel to be in logspace
    if logx:
        locs = np.arange(strat.lb[0], strat.ub[0])
        ax.set_xticks(ticks=locs)
        ax.set_xticklabels(2.0 ** locs)

    ax.plot(x[y == 0, 0], x[y == 0, 1], "ro", alpha=0.7, label="Nondetected trials")
    ax.plot(x[y == 1, 0], x[y == 1, 1], "bo", alpha=0.7, label="Detected trials")

    if target_level is not None:  # plot threshold
        mono_grid = np.linspace(strat.lb[1], strat.ub[1], num=gridsize)
        context_grid = np.linspace(strat.lb[0], strat.ub[0], num=gridsize)
        thresh_75, lower, upper = get_lse_interval(
            modelbridge=strat.modelbridge,
            mono_grid=mono_grid,
            target_level=target_level,
            cred_level=cred_level,
            mono_dim=1,
            n_samps=500,
            lb=mono_grid.min(),
            ub=mono_grid.max(),
            gridsize=gridsize,
        )
        ax.plot(
            context_grid,
            thresh_75,
            label=f"Est. {target_level*100:.0f}% threshold \n(with {cred_level*100:.0f}% posterior \nmass shaded)",
        )
        ax.fill_between(
            context_grid, lower, upper, alpha=0.3, hatch="///", edgecolor="gray"
        )

        if true_testfun is not None:
            true_f = true_testfun(grid).reshape(gridsize, gridsize)
            true_thresh = get_lse_contour(
                norm.cdf(true_f), mono_grid, level=target_level, lb=-1.0, ub=1.0
            )
            ax.plot(context_grid, true_thresh, label="Ground truth threshold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)
