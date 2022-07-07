---
id: gp_intro
title: A brief introduction to Gaussian Process active learning
---

## A brief introduction to Gaussian process models
The default model in AEPsych is the Gaussian Process (GP) classification model, a nonparametric function approximator that assumes the latent percept is smoothly varying with any number of stimulus configuration dimensions. More formally, the default generative model in AEPsych is:

$$
y \mid f \sim \mathcal{B}(\sigma(f)),\quad \mathbf{f}\mid x \sim \mathcal{N}\left(m, \Sigma\right),
$$

where \\(\Sigma\\)  is a \\(N \times N\\)  covariance (Gram) matrix whose \\((n,n')\\) th entry is given by a kernel function \\(\kappa(\mathbf{x}_n,\mathbf{x}_n')\\) evaluated over the \\(n\\)th and \\(n'\\)th stimuli, \\(N\\)  is the total number of stimuli sampled, \\(m\\)  is a constant (0 by default), and \\(\sigma(\cdot)\\)  is a link function mapping from real numbers to probabilities (the standard Normal CDF or *probit* by default).

Importantly, the “Gaussian” part of GPs is not a standard psychophysical noise assumption, but rather a technical assumption about the joint distribution of latent function observations. AEPsych supports a variety of psychophysical noise models including the Gaussian, Logistic, and Gumbel. The noise distribution appears in the expression above as its CDF \\(\sigma(\cdot)\\).

GPs are nonparametric in the sense that they do not use a parametric form to describe the psychometric field. Instead, the field is described implicitly by its values at any set of points and their covariance. This means that we take a separate interpolation step to extract estimates such as thresholds or JNDs.

## Model-based active learning with GPs
We can use a model to determine which point to sample next. To do so, we define an "acquisition function", which assigns a score to each point in the parameter space that tells us how informative sampling at that point would be for achieving our goals. For example, if our goal is to estimate the psychometric function at every point in the space, a simple acquisition function could be given by the current uncertainty over the function at each point. By always sampling at the points where the uncertainty is highest, we may be able to get an accurate model in fewer trials than by just sampling the space randomly.

In practice, however, creating new acquisition functions is an active area of research because model uncertainty often interacts with data noise in nontrivial ways (especially in psychophysics) -- for example, high-uncertainty points for psychonmetric function estimation are often close to where the response probability is 0.5, which is where the noisiest responses are, so sampling in other locations may be more effective. Furthermore, focusing on local uncertainty for choosing points may be less effective than considering how observing one location might affect uncertainty over the entire psychometric field. As a result, AEPsych contains a growing number of state of the art acquisition functions, and we have some recommendations for common workflows:

* If you are interested in threshold (level set) estimation, we recommend using the GlobalMI acquisition function. It attempts to sample at the point that is most informative about the threshold location over the full range of non-intensity variables when accounting for the possible human responses to the query.
* If you are interested in JNDs or global function estimation, we recommend BernoulliMCMutualInformation (also known as BALD in the literature) for smaller experiments (1-2d). BALD is similar to standard mutual-information-based strategies from adaptive psychophysics, and we find that it empirically performs poorly above 2-3d, so we recommend falling back to quasi-random search in higher dimensions.
* While pure optimization is not a common psychophysics task, if you would like to use AEPsych for tuning something based on human preferences, we recommend qNoisyExpectedImpprovement for acquisition.
