---
id: gp_intro
title: A brief introduction to Gaussian Process active learning
---

# The core AEPsych ingredients: models and acquisition functions

In AEPsych, we decompose the active learning problem into two questions: [1] what is the model we are using for the data? and [2] how are we using this model to select the next point to measure? The example configs make reasonable choices for both items, but it is important to remember that AEPsych is not a single monolithic method, but rather a platform for constructing model-based active learning algorithms.

The perspective on models in AEPsych is strongly influenced by [generalized linear models (GLMs)](https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab). In a GLM, there is a latent linear function that is transformed through a link function, and a likelihood distribution that describes how data is generated given the latent function taken through the link. The function describing the latent psychometric or preference function is what [psignifit](https://psignifit.sourceforge.net/PSYCHOMETRICFUNCTIONS.html) calls a _core_, and the link is what psignifit calls the _sigmoid_. In AEPsych, we usually use a flexible nonparametric Gaussian Process (GP) model for the latent function, but other options exist and are described below. For the psychophysics domain, the likelihood in AEPsych is typically Bernoulli, with a Probit (Gaussian CDF) link, but a few other options exist (also described below). For more on generalized GPs, see [Chan & Dong 2011](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr11-ggpm.pdf) and [Wang & Shi 2014](https://www.tandfonline.com/doi/abs/10.1080/01621459.2014.889021).

# Models in AEPsych

For a model with a GP prior, identity link, and Gaussian observation likelihood, the posterior is available in closed form. For all other models, AEPsych relies on [variational inference](https://arxiv.org/abs/1601.00670), which solves an optimization problem to construct approximate posteriors quickly enough for human-in-the-loop usage (for the specific approaches AEPsych benefits from, see [Hensman et al. 2015](https://proceedings.mlr.press/v38/hensman15.pdf) and the [BoTorch docs](https://botorch.org/api/fit.html#botorch.fit.fit_gpytorch_mll)). Every model class in [`AEPsych.models`](https://aepsych.org/api/models.html) is a subclass of either the variational or non-variational GP class, with some specification of likelihood and link (which we call `Objective` in the code to match the BoTorch terminology). For example, `GPBetaRegressionModel` is identical to instantiating `GPClassificationModel` with `likelihood=BetaLikelihood()`.

This section discusses the models AEPsych has for the (latent) psychometric or preference function. Below, we discuss the links and likelihoods. Models in AEPsych operate in various points of the bias-variance (or equivalently, prior-data) tradeoff. More flexible models are useful when you know fewer things about your data generating process, but their larger hypothesis space means they require more data. On the other hand, less flexible models build more structure in, reduce the size of the hypothesis space, and thereby require less data. In approximate order from most to least flexible, the AEPsych models are:

* A GP model with the Radial Basis Function (RBF) kernel over all dimensions.
* A GP model with the RBF kernel and monotonicity imposed on intensity dimensions via posterior projection or derivative constraints.
* A "semi-parametric" model that has a classical parametric form for the intensity dimension, and GP priors on the model parameters.


An earlier and more restrictive approach to constraining psychometic field intensity dimensions comes from [Song et al. 2017](https://link.springer.com/article/10.3758/s13414-017-1460-0), who decompose the GP kernel additively into an RBF kernel over context dimensions, and a linear kernel over the context dimension. That model assumed that the psychometric function has the same linear appearance over the whole space, and is shifted nonlinearly based on context. At least when using variational inference, [Owen et al. 2021](https://arxiv.org/abs/2104.09549) and [Keeley et al. 2023](https://arxiv.org/abs/2302.01187) showed that it under-performs the semi-parametric models in the small-data regime, and the full RBF models in the larger-data regime, so while AEPsych implements it, we do not recommend it for usage currently.

In addition, AEPsych provides a model for pairwise data (discussed below), and a way of constructing other custom kernels via kernel factory functions (discussed in the [API docs](https://aepsych.org/api/factory.html)).

## Conventional RBF-GP model

The default latent model in AEPsych is the Gaussian Process (GP) model, a nonparametric function approximator that assumes the latent percept is smoothly varying with any number of stimulus configuration dimensions. More formally, the generative model is:

$$
\mathbf{f}\mid x \sim \mathcal{N}\left(m, \Sigma\right),
$$

where $\Sigma$  is a $N \times N$  covariance (Gram) matrix whose $(n,n')$ th entry is given by a kernel function $\kappa(\mathbf{x}_n,\mathbf{x}_n')$ evaluated over the $n$th and $n'$th stimuli, $N$  is the total number of stimuli sampled, $m$  is a constant (0 by default). The most common choices for the kernel function are the (RBF) and Matérn, and AEPsych defaults to the former (in informal testing we have not noticed a difference between those two choices). The RBF kernel is defined as:

$$
\kappa(\mathbf{x}_n,\mathbf{x}_n') = v \exp \left(-\frac{1}{2}(\mathbf{x}_n,\mathbf{x}_n')^{T}\Theta^{-2}(\mathbf{x}_n,\mathbf{x}_n')\right)
$$

where $v$ is a positive _outputscale_ parameter and $\Theta$ is a diagonal matrix of per-dimensions _lenghtscales_. These _hyperparameters_ are estimated from data using [informative priors ](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale) that disprefer overly flat psychometric functions. The outputscale will scale the overall function, while the lengthscales measure the distance at which points are expected to be correlated (smaller lengthscales imply more "wiggly" functions). Estimating per-dimension lengthscales is known as Automatic Relevance Determination (ARD) and is a way of letting the model automatically determine which dimensions are more influential on the predicted outcome (in the sense that the outcome varies more rapidly with these parameters).

Importantly, the “Gaussian” part of GPs is not a standard psychophysical noise assumption, but rather a technical assumption about the joint distribution of latent function observations. AEPsych supports a variety of psychophysical noise models, as discussed in the section on link functions below. GPs are nonparametric in the sense that they do not use a parametric form to describe the psychometric field. Instead, the field is described implicitly by its values at any set of points and their covariance. This means that we take a separate interpolation step to extract estimates such as thresholds or JNDs.

**Usage Advice**: A GP model with the RBF kernel is _universal_ in the sense that in the limit of data, it can recover any functions, so it is a good choice if you have no a priori assumptions about the shape of the psychometric field and you would have otherwise used the method of constant stimuli or grid search. Because of the inherent smoothness assumption of the RBF kernel, this model performs most poorly if there are discontinuities or other very sharp transitions in the psychometic field. Because the RBF kernel is _stationary_, it assumes the covariance between any two points is only a function of their distance in parameter space, so if the characteristic smoothness of the psychometric field changes rapidly this model will likewise struggle. If you can make stronger assumptions about your data generating process that matches one of the more restricted models in AEPsych, you should use that model, as it will require less data. All likelihoods and links in AEPsych are compatible with this prior (for links to specific model code, see below in the discussion of likelihoods).

## RBF-GP model with monotonicity constraints

A slightly stronger assumption that one can make is that one or more dimensions of your problem are monotonically increasing. This is common in psychophysics problems where there is an intensity dimension such as contrast, but less common in preference learning. In practice, we found that monotonicity is very easy to learn for the vanilla RBF model (above) over most of the space, but that without any constraints such models will fail to have detection probabilities saturate to exactly 0 or 1 at the edges of the space. There are a two ways of imposing monotonicity on GPs in the literature: one is to somehow constrain the derivatives of the function (via priors or searching over a limited domain). The second is to project a conventional GP posterior (as above) to the nearest monotonic one. AEPsych implements both methods, as `MonotonicRejectionGP` and `MonotonicProjectionGP`.

**Usage Advice**: While derivative-constrained GPs are implemented in AEPsych, we found them to be too slow for human-in-the-loop usage while not providing substantial performance benefit. If monotonicity is needed in your setting, the monotonic projection GP is essentially a post-fitting step that will give you a (near-)monotonic psychometric function for plotting and prediction. Informally we suspect that this model is universal over monotonic functions, so should be fairly safe to use for standard psychometric problems when minimal prior knowledge is available.

## Semi-parametric model with GP priors
In many psychophysics settings, we have reason to assume that conventional intensity dimensions (contrast, brightness, etc) follow a classical psychometric function with a slope and intercept taken through a sigmoid. [Keeley et al. 2023](https://arxiv.org/abs/2302.01187) suggested that this means we can define GP priors directly on the parameters of such a psychometric function, where those GPs are a function of the remaining non-intensity dimensions, and showed that such a model can be substantially more performant than the conventional RBF-GP in the small-data regime. This model admits a few additioanl advantages, notably the ability to directly extract slopes (JNDs) and intercepts (thresholds) from the model without an interpolation step and ability to sample actively based on only the threshold posterior (for threshold estimation problems). AEPsych implements both the full semi-parametric model as `SemiParametricGPModel`, and an approximate model that derives a so-called "psychophysics kernel" based on the semi-parametric model while retaining the multivariate Gaussian form of the GP (as `HadamardSemiPModel`).

**Usage Advice**: The SemiP models pay for their efficiency by having a very strong constraint on the psychometric function in the intensity dimension. T use the SemiP models, you should be very sure that your intensity dimension follows a classical psychometric function, and that your intensity dimension is in the correct units.

## Pairwise models
Often in both psychophysics and preference learning participants make binary responses based on pairs of stimuli (e.g. "is X preferred to Y", or "is X louder than Y"). For this setting, AEPsych provides a pairwise classification model which is a thin wrapper around the [pairwise preference GP](https://botorch.org/tutorials/preference_bo) in BoTorch, implemented as [`PairwiseProbitModel`](https://aepsych.org/api/models.html#aepsych.models.PairwiseProbitModel). There are no non-classification and non-probit-link pairwise models in AEPsych currently.

**Usage Advice**. This model is currently the only option for paired data in AEPsych. Paired data generally provides more information about the shape of the psychometric function than single-stimulus data, and as such yields faster tuning/optimization runs and faster ability to characterize the full function at super-threshold levels. Intuitively, this is since each observation is informative about two rather than one location in stimulus space. However, paired data can only constrain the latent function up to a constant shift, so at least some single-stimulus observations are needed to reference the function and constrain that shift.

# Link functions and likelihoods in AEPsych
To get a full model, we need to combine one of the models above for the latent psychometric field with a likelihood through a link function. AEPsych supports a few different likelihoods:

* **Binary**: for yes/no or other kinds of [0,1] responses, AEPsych provides a Bernoulli likelihood with Probit, Logit, or Gumbel links.
* **Percentage**: for outcomes bounded between 0 and 1, AEPsych provides a Beta likelihood with a Logit link with a learned scale.
* **Ordinal**: for ordered discrete outcomes (e.g. likert scales)), AEPsych provides a Categorical Likelihood with a Probit, Logit or Gumbel link and learned cutpoints.
* **Continuous**: for unconstrained continuous outcomes, AEPsych provides a Gaussian likelihood with the identity link.

These are described in greater detail below:

## Models for binary data

Given the latent itnensity function $\mathbf{f}(x)$, the likelihood for a binary observation in AEPsych is $y \mid x \sim Bernoulli(\sigma(\mathbf{f}(x))$, where $\sigma$ is a Probit, Logit, or Gumbel sigmoid link function. In the psychophysics setting, these links correspond to the CDF of the sensory noise distribution (Gaussian, Logistic, or Log-Weibull).

**Usage Advice** For psychophysics applications, we recommend the Probit as a default, as it is most anchored in classical psychophysics theory. For other applications, the Logistic is more standard and can be faster numerically. However, ultimately this is a choice that should be set by cross-validation or a priori theoretical determination. This is implemented as [`GPClassificationModel`](https://aepsych.org/api/models.html#aepsych.models.gp_classification.GPClassificationModel) for the RBF kernel, [`MonotonicRejectionGP`](https://aepsych.org/api/models.html#aepsych.models.monotonic_rejection_gp.MonotonicRejectionGP) for the GP-RBF model with monotonicity constraints, [`SemiParametricGPModel`](https://aepsych.org/api/models.html#aepsych.models.SemiParametricGPModel) for the semi-parametric model, or [`HadamardSemiPModel`](https://aepsych.org/api/models.html#aepsych.models.HadamardSemiPModel) for the approximate-MVN semi-parametric model.

## Models for percentage data
For percentage outcomes or other outcome values between 0 and 1, the likelihood is $y \mid x \sim Beta(\sigma(f)s, (1-\sigma(f)s))$. This is a scale-mixture parameterization of the Beta distribution, where the mixture is given by the GP through a logistic sigmoid link, and the scale is a learned hyperparameters. No other links are supported (but we accept pull requests!).

**Usage Advice** This is uncommon in psychophysics applications, but can be useful in any sort of setting with continuous bounded outcomes, especially if there are observations close to the bounds (otherwise the regular continuous outcome model may be sufficient and run faster). An example application is a preference learning setting when users give scores on a 0-100 scale. This model is implemented as [`GPBetaRegressionModel`](https://aepsych.org/api/models.html#aepsych.models.gp_classification.GPBetaRegressionModel) for the RBF prior. For other priors, simply pass the `likelihood=BetaLikelihood()` configuration option to one of the models for binary data discussed above.

## Models for ordinal data
For ordered discrete (ordinal / likert) data, we follow the parameterization given by [Chu & Ghahramani 2005](https://www.jmlr.org/papers/volume6/chu05a/chu05a.pdf), which combines a sigmoid link and learned cutpoints. We review it here from a psychophysics framing, which may be useful for perceptual researchers. In standard psychophysics theory, we have two stimuli, $x_1$ and $x_2$, and we assume the latent representations of both are corrupted by noise to yield latent percepts:
$$
\begin{aligned}
\widetilde{f}(x_1) &= f(x_1) + \epsilon_1, \epsilon_1 \sim \mathcal{N}(0, \sigma)\\
\widetilde{f}(x_2) &= f(x_2) + \epsilon_2, \epsilon_2 \sim \mathcal{N}(0, \sigma),
\end{aligned}
$$
where under Weber's law we expect $f(x)$ to be the log function but in AEPsych we assume it is more general. For symmetric noise distributions, this is equivalent to saying $f(x_n) = \widetilde{f}(x_n) + \epsilon_n$, which is equivalent to the following model:

$$
\begin{aligned}
f(x_1) &\sim \mathcal{N}(\widetilde{f}(x_1), \sigma^2)\\
f(x_2) &\sim \mathcal{N}(\widetilde{f}(x_2), \sigma^2).
\end{aligned}
$$

Then when we ask the participant to respond based on which stimulus is stronger, we ask them to respond yes if $f(x_1)>f(x_2)$ and respond no otherwise. By definition,

$$
\begin{aligned}
p(f(x_1) > f(x_2)) &=p(f(x_1) - f(x_2) > 0)\\
&= \Phi(f(x_1)-f(x_2)).
\end{aligned}
$$

In detection experiments, we treat $f(x_2)$ as a constant, and for an $f(\cdot)$ that can model shifts it can just be zero, yielding $z(x\mid f) = \Phi(f(x))$. We often care about the marginal predictive probability $z(x) = \mathbb{E}_f(z(x\mid f)$, which is given by the following expression:
$$
z(x) = \Phi\left(\frac{\mu(x)}{\sqrt{1+\sigma(x)^2}}\right),
$$
where the mean and variance are the mean and variance of $f(x)$ (i.e. for the prior $\mu(x)=\widetilde{f}(x)$).

Now we extend this for multiple stimuli. Suppose that:
$$
\begin{aligned}
\widetilde{f}(x_1) &= f(x_1) + \epsilon_1, \epsilon_1 \sim \mathcal{N}(0, \sigma^2)\\
\ldots\\
\widetilde{f}(x_n) &= f(x_n) + \epsilon_n, \epsilon_n \sim \mathcal{N}(0, \sigma^2),
\end{aligned}
$$
for $n$ stimuli, and equivalently $f(x) \sim \mathcal{N}(\widetilde{f}(x), \sigma^2)$ (i.e.\ it's the same exact model). Then we ask the participant to make one of $n$ responses corresponding to the strength of the stimulus. We assume that the participant has a set of internal criteria $\{d_1, \cdots, d_{k+1}\}$ subdividing their internal perceptual intensity space into $k$ regions. Assuming each region in the space must be assigned to some region, we set $d_1=-\infty$ and $d_{k+1} = \infty$. Then the probability that a participant picks a rating is the probability that the noisy internal percept falls into the appropriate bucket, i.e. $p(y=k\mid x) = p(d_k < f(x) \le d_{k+1})$. This is simply the proportion of $f(x)$ that falls between $d_k$ and $d_{k+1}$, i.e.\ it is the difference between the CDF of $f(x)$ evaluated at $d_{k+1}$ and $d_k$:
$$
z_k(x\mid f) := p(d_k < f(x) \le d_{k+1}) = \Phi(d_{k+1}-f(x)) - \Phi(d_{k}-f(x)).
$$
Supposing that we want the marginal response probability as before, due to the linearity of expectation it is:
$$
z_k(x) = \Phi\left(\frac{d_{k+1}-\mu(x)}{\sqrt{1+\sigma(x)^2}}\right) - \Phi\left(\frac{d_{k}-\mu(x)}{\sqrt{1+\sigma(x)^2}}\right),
$$

**Usage Advice**: Likert-scale data is common in user experience and other preference-oriented studies, but we believe based on the framing above that it has some promise in psychophysics as well. For example, Ordinal-response models could be used to model subjective intensity judgments such as how bright or loud a stimulus is on a likert scale. Note that the ordinal model is unconstrained w.r.t. a shift (similarly to the Pairwise model) and also an arbitrary multiplicative scaling of the function. This model with the RBF kernel is exposed as [`OrdinalGPModel`](https://aepsych.org/api/models.html#aepsych.models.OrdinalGPModel). For other likelihoods and links, you would need to subclass from one of the classification models above, and pass in a kernel without an outputscale to account for the scale-invariance. Currently such models can be constructed in python code but not using the AEPsych configuration system -- we accept pull requests!

## Models for continuous data
For continuous data, we use a Gaussian likelihood with the identity link, i.e. $y \mid x \sim \mathcal{N}(\mathbf{f}(x)+ \sigma^2)$ with a learned variance $\sigma^2$. This is analogous to the standard GP regression model typically used in Bayesian Optimization.

**Usage Advice**: this model can be suitable for continuous psychophysics and preference ratings, and can be a convenient way to integrate continuous tuning into a project already using the AEPsych client-server interface. It is implemented as [`GPRegressionModel`](https://aepsych.org/api/models.html#aepsych.models.GPRegressionModel) for the RBF-GP setting. It is not implemented or tested with the monotonic or semi-parametric priors, though it should be straightforward to do so if it that is needed for your application and we are happy to review implementation plans and PRs on this front. If your continuous ratings are on a likert scale, we recommend the Ordinal model instead. If your continuous ratings are over a limited range (e.g. 0-100) and you expect a nontrivial amount of them to be close to the boundaries, you should use the Beta regression model instead. If you purely need to apply Bayesian Optimization or continuous GP models without the other AEPsych psychophysics machinery or clients, you can also be well-served using the [Ax](https://ax.dev/) adaptive experimentation package (which AEPsych extends) directly.

## Model-based active learning in AEPsych
AEPsych is a platform for model-based experiment design, i.e. the use of a model (as discussed above) to determine the point to sample next. Currently, AEPsych defaults to _query-based_ active learning, which means that there is no fixed set of stimuli over which we are searching (as would be the case in _pool-based_ active learning more common in classical adaptive psychophysics). Rather, we can query any stimulus in the space. To determine where to sample next, we define an _acquisition function_ which, given a model, assigns a value to each point in the parameter space that tells us how useful sampling at that point would be for achieving our goals. By optimizing the acquisition function w.r.t. the stimulus configuration, we find the best stimulus to sample next. We then sample this stimulus, update the model, re-optimize the acquisition function, rinse and repeat.

By way of example, if our goal is to estimate the psychometric function at every point in the space, a simple acquisition function could be given by the current uncertainty over the function at each point. By always sampling at the points where the uncertainty is highest, we may be able to get an accurate model in fewer trials than by just sampling the space randomly. In practice, however, creating new acquisition functions is an active area of research because model uncertainty often interacts with data noise in nontrivial ways (especially in psychophysics) -- for example, high-uncertainty points for psychonmetric function estimation are often close to where the response probability is 0.5, which is where the noisiest responses are, so sampling in other locations may be more effective. Furthermore, focusing on local uncertainty for choosing points may be less effective than considering how observing one location might affect uncertainty over the entire psychometric field. As a result, AEPsych contains a growing number of state of the art acquisition functions.

From a technical perspective, there are two categories of acquisition functions in AEPsych:

1. Monte Carlo (MC) acquisition functions. These acquisition functions can be evaluated by sampling from the model posterior. In the simplest case we sample from the latent perceptual field posterior, but we can also sample from other posteriors such as the response probability posterior, or response threshold posterior. The most common acquisition functions of this kind is BALD, or information maximization w.r.t. the response probability posterior (aka [`BernoulliMCMutualInformation`](https://aepsych.org/api/acquisition.html#aepsych.acquisition.mutual_information.BernoulliMCMutualInformation) in AEPsych), or BALV, or variance maximization (aka [`MCPosteriorVariance`](https://aepsych.org/api/acquisition.html#aepsych.acquisition.mc_posterior_variance.MCPosteriorVariance) in AEPsych). [Keeley et al. 2023](https://arxiv.org/abs/2302.01187) additionally introduced `ThresholdBALV`, which is BALV defined over the threshold rather than probability posterior for the semi-parametric GP models in AEPsych. To use `ThresholdBALV`, use `MCPosteriorVariance` with the `objective=[SemiPThresholdObjective](https://aepsych.org/api/acquisition.html#aepsych.objective.SemiPThresholdObjective)` option. Because they only require sampling from the posterior, they can be used with any model which admits sampling from the posterior.
2. Analytic acquisition functions. These acquisition functions require an explicit multivariate normal (MVN) posterior to be evaluated. They can often be faster to evaluate, and enable efficient computation of  _lookahead_ approaches, which compute the acquisition in expectation of the posterior as it will be given the next observation. As with MC acquisition functions, they can be evaluated w.r.t. the current or lookahead function posterior (as in [Zhao et al. 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/50d2e70cdf7dd05be85e1b8df3f8ced4-Abstract.html), the lookahead level set (threshold) posterior, as described in [Letham et al. 2022](https://arxiv.org/abs/2203.09751), or any other posterior that can be approximated as MVN (e.g. the full semi-parametric posterior, with the MVN approximation taken after posterior inference.

For more on analytic and MC acquisition functions, you can check the [BoTorch docs](https://botorch.org/docs/acquisition), and AEPsych supports all BoTorch acquisition functions, as well as a growing number of acquisition functions developed specifically for human psychophysics. From an application perspective, acquisition functions in AEPsych could be sorted based on their goals:

* **Optimization.** While pure optimization is not a common psychophysics task, it is very common in preference learning. If you would like to use AEPsych for optimizing something, we recommend `qNoisyExpectedImpprovement` for acquisition for all but the continuous-outcome Gaussian-noise model, where we recommend `qKnowledgeGradient`. The former samples the point most likely to be better than the current best sample observed, where the current best is also estimated from the current (noisy) model. The latter, a lookahead approach, exploits the MVN assumption to sample the point most likely to be better than the current best known function value (even if it had not been observed). Both are directly imported into AEPsych from [BoTorch](https://botorch.org/).
* **Level-set estimation**, or threshold estimation, is only supported in AEPsych for Bernoulli observation models. If you are interested in threshold (level set) estimation , we recommend using the [`GlobalMI`](https://aepsych.org/api/acquisition.html#aepsych.acquisition.GlobalMI) or [`EAVC`](https://aepsych.org/api/acquisition.html#aepsych.acquisition.EAVC) acquisition function. Both were shown to be similarly competitive by [Letham et al., 2022](https://arxiv.org/abs/2203.09751) but in practice we found sometimes one works better than the other in specific applications. `GlobalMI` attempts to sample at the point that is most informative about the threshold location over the full range of non-intensity variables when accounting for the possible human responses to the query. `EAVC` attempts to maximize the expected absolute volume change between the sublevel-set volume (i.e. volume below threshold) before and after taking the next observations, in expectation over the value of the next observation. Both `EAVC` and `GlobalMI` are lookahead acquisition functions, and theoretically require a probit link, though in practice they seem to work with other links as well (using one of them with a non-probit link implicitly uses a probit link for computing the acquisition function).
* **Global uncertainty reduction**. If you are interested in JNDs or global function estimation, the best approach is likely `GlobalMI` or `EAVC` with `acqf_kwargs={lookahead_type:'posterior'}` in instantiating your generator, or `lookahead_type = posterior` under your generator section in the AEPsych configuration. This option is not not yet extensively tested -- a safe but less efficient option is `BernoulliMCMutualInformation` (also known as BALD in the literature) for smaller experiments (1-2d). BALD is similar to standard mutual-information-based strategies from adaptive psychophysics, and we find that it empirically performs poorly above 2-3d, so we recommend falling back to quasi-random search in higher dimensions. We would be grateful for reports of successful use of `GlobalMI` or `EAVC` for global uncertainty reduction in the wild.
