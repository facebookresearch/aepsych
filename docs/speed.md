---
id: speed
title: Active Learning Speedups
---

This page provides documentations and our recommendations for speeding up AEPsych during
active learning. We detail features built into AEPsych intended to allow AEPsych's
server to respond faster during an experiment as well as our recommendations on config
settings that affect active learning speed that may change results.

Psychophysics experiments may have participants responding to a trial in less than a
second after the trial onset. When using AEPsych, if it takes the server too long to
respond, the time it takes to complete an experiment can be very long and ultimately be more
costly. Further, longer experiments may cause participants to become fatigued, yielding
worse results. Thus, speeding up an experiment can yield significant benefits.

<h2>Speed-up Features</h2>

We implemented multiple features to allow speeding up AEPsych's server in response to
messages. These features can be used together and have different
effects on the effectiveness of the AEPsych response speed.

<h3>GPU support</h3>

The `GPClassification` and `GPRegressionModel` both have support to run on the GPU. Models
that subclass these models should also have GPU support. To get a model running on the
GPU, the `use_gpu` option for the model should be set. By default, the models will not
use a GPU (even if a GPU is available).

```ini
[opt_strat]
model = GPClassificationModel
generator = OptimizeAcqfGenerator

[GPClassificationModel]
use_gpu = True # turn it on with any of true/yes/on, turn it off with any of false/no/off; case insensitive
```

This will cause the model fitting during active learning to use the GPU. **With the
amount of data that will typically be in a live experiment, using a GPU to fit the model
will not result in a speed up and may incur a slowdown instead**.

However, there may be cases (e.g., high dimensionality, many parameters, many trials,
or pos-hoc analysis with a lot of data)
where using the GPU for model fitting will make it faster. This is also hardware
dependent. If speed is a concern, it is worth testing to see if using a GPU will speed
up model fitting. The log will provide timing to help decide whether using a GPU for
model fitting is worth it.

Generators can also use the GPU. This is usually the most time-consuming part of
responding to an ask message to the server. Using a GPU here will typically provide at
least a modest speedup (if not 2-5x faster).

Currently, the `OptimizeAcqfGenerator` and any available acquisition function will
support using the GPU. As in the models, the `use_gpu` option in the config should be
set for the generator. By default, the generators will not use a GPU (even if a GPU is
available).

If the server cannot find a GPU even though GPUs were requested for either models or
generators, it is likely that PyTorch cannot access the GPUs. Reinstalling PyTorch
with GPU support should fix this.

```ini
[opt_strat]
model = GPClassificationModel
generator = OptimizeAcqfGenerator
acqf = MCLevelSetEstimation

[OptimizeAcqfGenerator]
use_gpu = True # turn it on with any of true/yes/on, turn it off with any of false/no/off; case insensitive
```

The time it takes to generate a point is dependent on the acquisition function. For the
most common use-case of threshold estimation, the MCLevelSetEstimation acquisition
function is often the default choice as it is typically very fast. However, it is not
the state-of-the-art in terms of active learning efficacy. `EAVC` and `GlobalMI` are
often more efficient at identifying thresholds for complex or high-dimensional problems
as they are less likely to sample at the edges of the space, but they are also slower
at trial generation. If the generator is run on the GPU, both `EAVC` and `GlobalMI` yield
comparable speeds as `MCLevelSetEstimation`, while suggesting better points to test for
active learning.

On a workstation with an AMD Ryzen Threadripped PRO 3795WX 32-Cores CPU and a NVIDIA
GeForce RTX 3080 GPU, these are the speed benchmarks on a simple GPClassificationModel
fit on 3-dimensional Sobol points.

| Fitting |      n=10     |      n=50     |     n=100     |
|---------|:-------------:|:-------------:|:-------------:|
| CPU     |     0.12s     |     0.46s     |     0.77s     |
| GPU     | 0.27s (2.13x) | 0.93s (2.02x) | 1.33s (1.73x) |

Fitting simple models with the magnitude of data within an active learning experiment
shows slowdowns with the GPU.

However, generating points with different acquisition functions can be faster.

| MCLSE |      n=10     |      n=50     |     n=100     |
|-------|:-------------:|:-------------:|:-------------:|
| CPU   |     0.16s     |     0.64s     |     1.06s     |
| GPU   | 0.35s (2.24x) | 0.91s (1.43x) | 1.64s (1.54x) |

The MCLevelSetEstimation acquisition function is typically the fastest and using the
GPU with it causes some slowdown.

| EAVC |      n=10     |      n=50     |     n=100     |
|------|:-------------:|:-------------:|:-------------:|
| CPU  |     1.44s     |     2.74s     |     3.26s     |
| GPU  | 0.41s (0.28x) | 1.50s (0.55x) | 1.78s (0.48x) |


| GlobalMI |      n=10     |      n=50     |     n=100     |
|----------|:-------------:|:-------------:|:-------------:|
| CPU      |     1.59s     |     2.78s     |     3.60s     |
| GPU      | 0.63S (0.40x) | 1.72s (0.78x) | 1.82s (0.56x) |

Both EAVC and GlobalMI are usually better acquisition functions, allowing for more
efficient active learning demonstrates significant speedups allowing them to be
comparable to MCLevelSetEstimation. Keep in mind these results are with a machine
that has a very powerful CPU and a typical GPU. It is likely that the differences
between a modestly powerful CPU and a typical GPU will be favor GPUs more often.

If possible, we recommend using the GPU only for the generator and the better
acquisition functions for active learning. It should be possible to confidently estimate
thresholds with fewer trials using better acquisition functions, therefore allowing
shorter experiments with little-to-no loss in modeling effectiveness. Again, it is worth
piloting experiments using the GPU and without the GPU for the generator with the experiment
hardware to double-check the effectiveness.

<h3>Refit Intermittently</h3>

By default, the model will be refit hyperparameters after every tell. While the fitting time
may not be the most time-consuming part, it is possible to shorten the AEPsych server response time
to asks by only refitting the hyperparameters model once every few asks. This does necessarily mean
that the model could be used to generate points without the entirety of the available
data during an experiment. This feature can be enabled by using the `refit_every` option
in a strategy's section. Regardless of what is set for this option, the model continues to be
conditioned on the data as it comes in.

```ini
[opt_strat]
generator = OptimizeAcqfGenerator
acqf = EAVC
model = GPClassificationModel
refit_every = 2 # A strictly positive integer
```
The `refit_every` will have the model only refit hyperparameters to the data every `n` data points. In
the above example, the model will only be refit every other tell, which halves the
overall fitting time across the whole experiment at the cost of the model not being completely optimized to the latest data.
two data points behind.

Refitting intermittently may be useful, especially in experiments with
many Sobol or manual trials before active learning, such that single trials are unlikely
to widely change the model fit. However, fitting intermittently may be bad for
exploration experiments where there may be relatively few trials for regions of the
parameter space.

<h3>Max Fit and Generating Time</h3>

It is possible to limit the time it takes to fit the model or generate points. While
this may result in suboptimal fits or suggested points, setting max times caps out how
long a participant may be waiting for a new trial to be generated.

Limiting max fitting time can be enabled with the `max_fit_time` option for a model.

```ini
[GPClassificationModel]
max_fit_time = 2.5 # Float in seconds
```

When `max_fit_time` is set, the AEPsych server calculates how many times the model can
be evaluated within the given time and limits the number of times the model can be
evaluated during the fit. This number is reported in the log as `maxfun`.

Limiting max point generation time can be enabled with `max_gen_time` option for a
generator.

```ini
[OptimizeAcqfGenerator]
max_gen_time = 2.5 # Float in seconds
```

When `max_gen_time` is set, the generation process has a timeout where if a point is
not chosen by the timeout, the best point at that time will be returned.

Both of these settings are soft constraints and may not be strictly respected.

Both of these maximum time settings may harm the active
learning loop, especially if either are set too low. Be careful when using these options
and examine the data after piloting to ensure that these times are not set too low.

<h2>Active Learning Tuning</h2>

There are many options that affect the time it takes for the AEPsych server to respond
to a message. These options can be tuned with speed-performance trade-offs. While using
the best options for each of these will likely produce better data, it may slow down the
active learning process sufficiently such that it is impractical in a real experiment. It is
worth piloting and analyzing the data to tune these options to best align with the
experiment's goals.

<h3>Inducing Points</h3>

When fitting approximate GP models (like the GPClassificationModel), using the entirety
of the data can be too costly. Instead, we distill the data down to inducing points for
variational inference. The number of inducing points ultimately determines how long a
model takes to fit. The more inducing points used the better the model will be but the
fitting time will also increase. Similarly, different inducing point selection
algorithms will result in different number of inducing points with varying levels of
how well the inducing points approximate the data.

By default, we set the maximum inducing points to 100 and use a Greedy Variance
Reduction algorithm implemented by BoTorch to select inducing points. This typically
results in far fewer than that 100 inducing points even with more than 100 data points,
thus yielding fast model fits. On very specific hardware when the number of data points
reaches a certain point (about 100), model fitting can slow down precipitously (x5-10
slower), if this does happen, please raise an issue on Github and we can look into it. This is a
very rare bug that only happens on specific hardware with specific array acceleration
libraries.

These settings can be modified in the model settings.

```ini
[GPClassificationModel]
inducing_size = 50 # This controls the maximum number of inducing points
inducing_point_method = kmeans++ # This controls the algorithm, can be pivoted_chol (for the default Greedy Variance Reduction), kmeans++, or all (just use all the data)
```

For even faster fits, the number of inducing points can be reduced. For better (but
slower) fits, the number of inducing points can be increased or other inducing point
selection algorithms can be used (e.g., `kmeans++`). Inducing point selection algorithms
other than Greedy Variance Reduction may result in better fits but will increase model
fitting time (especially with more data points/higher number of inducing points).

The rough heuristic for the number of inducing points to select is 50 for each
dimension, but this is a very rough heuristic that may be too high for simple parameter
spaces or too low for complex parameter spaces.

<h3>Acquisition Functions</h3>

Generating points is typically the most time-consuming portion of AEPsych generating a
response. By changing the acquisition function of the `OptimizeAcqfGenerator`, it is
possible to tune the performance of active learning.

The acquisition functions can be set in the generator options. There may also be
additional acquisition function settings to change the speed and effectiveness of the
acquisition function.

```ini
[OptimizeAcqfGenerator]
acqf = GlobalMI
```

In general, global lookahead functions (e.g. `GlobalMI`) yield the best results but take
more time (see above for using the GPU to accelerate these acquisition functions). Local
variants (e.g., `LocalMI`) can be faster but yield worse results. The commonly-used
MCLevelSetEstimation is very fast for threshold estimation but may yield less
informative points (which may require more trials to be run costing more time overall).

<h3>Fit to Recent Data</h3>

By default, models will be fit to all available data. It is possible to fit
on only some of the data, starting from the most recent. This is useful if the
responses are expected to change over time where the most recent data is more
informative but it can also limit the number of data points are used for fitting
(e.g., in very long experiments).

Given that this is only fitting to a subset of the data, this could yield worse active
learning results, but it could decrease fitting times significantly if many trials are
expected (e.g., starting with many Sobol generator or manual generator points). This
option can be set using the `keep_most_recent` option in a strategy.

```ini
[opt_strat]
model = GPClassificationModel
generator = OptimizeAcqfGenerator
acqf = EAVC
keep_most_recent = 50 # A strictly positive integer, keeping the 50 most recent points
```

In general, lowering the amount of data the model can fit on will weaken active learning
performance unless there's significant change in responses over time. However, with very
long experiments targeting a specific and reliable response probability, it may be worth it to only
use the most recent bit of data. As usual, it is worth piloting and tuning this option
if it is being used to test whether it significantly improves the server response time
while not harming (or improving) fits by the end.
