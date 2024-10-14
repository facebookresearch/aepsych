---
id: configs
title: Writing Config Files
---

While AEPsych is based on state of the art machine learning models, users are able to easily configure experiments by writing config files without having to understand any of the underlying code. Example configs can be found in the main repository and in the pre-release repository for common experimental paradigms. This page provides an in-depth explanation of these configs so that you can understand how to write your own from scratch.

You may wish to familiarize yourself with the workflow of psychophysics experiments and the [components of the AEPsych API](/docs/api_overview) before reading this tutorial. Also note that AEPsych is built on top of [Botorch](https://botorch.org/), so we may occasionally refer to those docs.

<h2>Config File Format</h2>
Config files are text files that specify the settings of an AEPsych experiment using the [INI format](https://en.wikipedia.org/wiki/INI_file). The files are broken into named sections, with each section specifying the settings of a different component of AEPsych. Here is an example of what a config file may look like:

```
[SectionName]
# Anything that comes after a '#' on a line is a comment
setting1 = 1 # this setting is set to a numerical value
setting2 = [a, b, c] # this setting is set to a list of strings

[SectionName2]
setting1 = 2 # this is a different section, so setting1 above is unaffected
```

<h1>Breaking Down an Example Config</h1>
To be more concrete, we will break down [one of the example config files](https://github.com/facebookresearch/aepsych/blob/main/configs/nonmonotonic_optimization_example.ini). This example config specifies a 2-dimensional optimization experiment. In other words, we want to find the values of our 2 parameters that maximize the probability of a binary outcome (e.g., the probability that the participant detected the stimulus). The comments explain how you can easily modify the file for your own experiment. Let's look at each of the sections so that we can understand them better.

<h3>common</h3>
```
[common]
parnames = [par1, par2] # names of the parameters
outcome_type = single_probit # we show a single stimulus and receive a binary outcome e
strategy_names = [init_strat, opt_strat] # the names we give to our strategies
```

The first section in the file is the `common` section. All AEPsych config files will have this section because it specifies global settings that all experiments use. Here is an explanation of each of these settings:

**`parnames`**: This is a list of parameter names. This example uses the generic names par1 and par2, but you can name your parameters whatever you would like. Note that this list should be the same length as lb and ub.

**`outcome_type`**: This is the type of outcome you will receive on each trial of your experiment. `single_probit` should be used for experiments where participants are shown a single stimulus and asked to make a binary choice, such as whether they detected the stimulus or not. `single_continuous` should be used for experiments where participants are shown a single stimulus and asked to provide a continous rating, such as how bright or loud the stimulus is.

**`strategy_names`**: This is a list of the data-collection strategies you plan on using in your experiment. Each named strategy in this list will receive its own section later in the config where we can specify the settings we want. This example follows a typical AEPsych paradigm where we first sample points in a quasi-random way to initialize our model, then we search for the optimal points using the model. Therefore we specify two strategies: `init_strat` (the initialization strategy) and `opt_strat` (the optimization strategy). Note that we chose these names to be mnemonic, but we could have named them whatever we wanted.

<h3>Parameter specific settings</h3>
```
[par1]
par_type = continuous
lower_bound = 0
upper_bound = 1

[par2]
par_type = continuous
lower_bound = 0
upper_bound = 1
```

Each parameter will have its own section based on the name defined in the `common` section.

**`par_type`**: This is the type of parameter it is, for now, this should just always be continuous.

**`lower_bound`**: This is the lower bound of this parameter.

**`upper_bound`**: This is the upper bound of this parameter.

<h3>init_strat</h3>
```
[init_strat]
min_asks = 10 # number of sobol trials to run
generator = SobolGenerator # The generator class used to generate new parameter values
This section specifies our first data-collection strategy. It is called init_strat because that is what we named our first strategy in strategy_names above. Here is an explanation of each of the settings:
```

**`min_asks`**: This is the number of trials that this strategy will run for. After 10 trials, the next strategy will begin.

**`generator`**: This specifies how this strategy will recommend points to sample. This strategy uses `SobolGenerator` to sample points quasi-randomly from the parameter space using the [Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence). These points will be used to initialize the model that will guide data-collection in the following strategy.  If for some reason we wanted to use Python's pseudo-random number generator instead of the Sobol sequence, we could have set the generator to `RandomGenerator`.

<h3>opt_strat</h3>
```
[opt_strat]
min_asks = 20 # number of model-based trials to run
refit_every = 5 # how often to refit the model
generator = OptimizeAcqfGenerator # The generator class used to generate new parameter values
model = GPClassificationModel # The model class
acqf = qNoisyExpectedImprovement # The acquisition function
```
This section specifies the model-based data-collection strategy. It is called opt_strat because that is what we named our second strategy in `strategy_names` above. Here is an explanation of each of the settings:

**`min_asks`**:This is the number of trials that this strategy will run for. After 20 trials, the experiment will complete.

**`refit_every`**:  On every trial, a new model is always fitted on the latest data, but the model's hyperparameters can be either initialized randomly and refit from scratch, or they can be initialized to the previous model's hyperparameters to save time. This setting determines how often the former happens instead of the latter. In this case, the model will be refit from scratch after every 5 trials. Refitting the mode from scratch often helps ensure that the best points will be sampled, but it also increases the amount of time it takes to generate new points. If trials are taking too long to run, try increasing this number.

**`generator`**: This specifies how this strategy will recommend points to sample. This strategy uses OptimizeAcqfGenerator, which uses an acquisition function in conjunction with the model to generate points. For pairwise experiments, you should use PairwiseOptimizeAcqfGenerator. Unlike SobolGenerator, OptimizeAcqfGenerator requires the strategy to have both a model and an acquisition function. If you do not specify them, you will receive an error. Refer to the generators folder for more options.
model: This specifies the model used to generate points. The model can also be used to analyze the data after the experiment is completed. This strategy uses a GPClassificationModel, a Gaussian process classifier.  Refer to the models folder for more options.

**`acqf`**: This specifies the acquisition function used to generate points. This strategy uses `qNoisyExpectedImprovement`, the recommended acquisition function for optimization experiments. If you wanted to do global exploration you could change this to `MCPosteriorVariance`, or if you wanted to do threshold finding you could change this to `MCLevelSetEstimation`. Refer to the documentation for A[EPsych acquisition functions](/api/aepsych.acquisition) and [Botorch acquisition functions](https://botorch.org/api/acquisition.html) for more options.

<h3>GPClassificationModel</h3>
```
[GPClassificationModel]
inducing_size = 100 # Number of inducing points for approximate inference.
mean_covar_factory = default_mean_covar_factory # the GP mean and convariance
```

This section specifies the settings for the GPClassificationModel used within opt_strat.  Here is an explanation of each of the settings:

**`inducing_size`**: This specifies the number of inducing points that are used in the Gaussian process model, which determine the quality of the approximate model posterior. A rule of thumb you can use is to set `inducing_size` to 50 times your number of dimensions.

**`mean_covar_factory`**: This determines the mean and covariance functions used in the Gaussian process model. Advanced users can specify their own factory functions to customize their models, but you should leave this alone unless you are sure you know what you are doing.

<h3>OptimizeAcqfGenerator</h3>
```
[OptimizeAcqfGenerator]
restarts = 10 # number of restarts for acquisition function optimization.
samps = 1000 # number of samples for random initialization of the acquisition function
```

This section specifies the settings for the `OptimizeAcqfGenerator` used in the model-based strategy. The restarts and samps settings determine how much time is spent finding the optimal values of the acquisition function (note that these settings correspond to `num_restarts` and `raw_samples` in Botorch's optimization function. Decreasing these values may increase the speed at which points are generated at the cost of decreasing the quality of the samples.

<h3>qNoisyExpectedImprovement</h3>
```
[qNoisyExpectedImprovement]
objective = ProbitObjective # The transformation of the latent function
```

This section specifies the settings for the `qNoisyExpectedImprovement` acquisition function used in the model-based strategy. The only setting that needs to be specified for this acquisition function is the `objective`, which we set to `ProbitObjective` because we have a binary outcome variable. Other acquisition functions may have other settings that can be specified.

<h2>Default Values</h2>
Some settings have default values and do not have to be explicitly set in the config. For example, the default values of restarts and samps within `OptimizeAcqfGenerator` are already 10 and 1000, respectively, so we could have left the `OptimizeAcqfGenerator` section out of the config, and its behavior would have been the same.

Similarly, `SobolGenerator` has a parameter called seed, which determines the random seed used to generate points. We did not include a `SobolGenerator` section in our config, so we used the default value of None, which means that a different random seed will be used each time the experiment is conducted. If we wanted to get the same Sobol points every time, we could specify a seed in the config like so:

```
[SobolGenerator]
seed = 123
```

To see all the settings that AEPsych components make use of, and their default values where applicable, refer to the [API Reference](/api).

You can also specify settings within the common section  to make them the default values used throughout the experiment. For example, adding the lines below  to the common section would make 5 the default number of trials and GPClassificationModel the default model for all strategies.
```
min_asks = 5
model = GPClassificationModel
```
<h2>Multiple Model-Based Strategies</h2>
AEPsych configs allow you mix-and-match as many strategies as you would like, and each strategy can have its own generator, model, and acquisition function. For example, the config below first does 5 Sobol trials, then 10 global exploration trials, and then finally 15 optimization trials.
```
[common]
parnames = [par1, par2]
outcome_type = single_probit
strategy_names = [sobol_strat, explore_strat, opt_strat]

[par1]
par_type = continuous
lower_bound = 0
upper_bound = 1

[par2]
par_type = continuous
lower_bound = 0
upper_bound = 1

[sobol_strat]
min_asks = 5
generator = SobolGenerator

[explore_strat]
min_asks = 10
generator = OptimizeAcqfGenerator
model = GPClassificationModel
acqf = MCPosteriorVariance

[opt_strat]
min_asks = 15
generator = OptimizeAcqfGenerator
model = GPClassificationModel
acqf = qNoisyExpectedImprovement

[MCPosteriorVariance]
objective = ProbitObjective

[qNoisyExpectedImprovement]
objective = ProbitObjective
```

<h2>Adding Models to Model-less Generators</h2>
Some generators, such as `SobolGenerator` and `RandomGenerator`, do not use models to generate points, so you do not have to specify a model in their strategies. However, there may be situations where you want to generate points using a model-less generator, but still fit a model. For example, if you have a very high-dimensional problem, it may be more efficient to sample the parameter space using `SobolGenerator` than to try to optimize an acquisition function with `OptimizeAcqfGenerator`, but a model is still required to analyze your data. In that case, you can specify a model in your strategy section, but you should set `refit_every` to be equal to `min_asks` so that you wait to fit the model at the end of the experiment and do not unnecessarily waste time fitting it while generating points (see the example below).

```
[common]
parnames = [par1, par2, par3, par4]
outcome_type = single_probit
strategy_names = [sobol_strat]

[par1]
par_type = continuous
lower_bound = 0
upper_bound = 1

[par2]
par_type = continuous
lower_bound = 0
upper_bound = 1

[par3]
par_type = continuous
lower_bound = 0
upper_bound = 1

[par4]
par_type = continuous
lower_bound = 0
upper_bound = 1

[sobol_strat]
min_asks = 20
refit_every = 20
generator = SobolGenerator
model = GPClassificationModel
```

<h2>Controlling the Timing of Trials (Experimental)</h2>
It is often important that each trial in a psychophysics experiment can be completed quickly so that the participant's and experimenter's time is not wasted. As noted elsewhere in this document, there are many settings that can be tweaked to make point generation and model fitting faster, but determining what values these settings should take to meet your timing constraints can take a lot of trial and error. To eliminate this guesswork, we have introduced two settings to explicitly control trial timing. You can set the maximum amount of time, in seconds, that should be spent generating points using max_gen_time under `OptimizeAcqfGenerator`, and you can set the maximum amount of time, in seconds, that should be spent fitting the model using `max_fit_time` under `GPClassificationModel`. These settings are considered experimental because they are still undergoing testing, but our initial findings indicate that they can be used to dramatically decrease computation time with minimal impact on results.
