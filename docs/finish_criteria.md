---
id: finish_criteria
title: Advanced Strategy Configuration
---

This page provides an overview of some new configuration options for AEPsych strategies that give you more control over which points will be sampled and when the strategy will end. You should have some familiarity with the [AEPsych API]((/docs/api_overview)) and [basic configs](/docs/configs) before reading this note.

<h2>Ending Strategies Based on the Amount of Data Collected</h2>
The [example configs](https://github.com/facebookresearch/aepsych/blob/main/configs/) tell AEPsych when to stop a strategy by specifying `min_asks` in a strategy's config. This setting tells AEPsych to end the strategy after it has generated a certain number of points (in other words, after `server.ask` has been called a certain number of times on the strategy). This isn't always ideal, though. There may be scenarios where you ask the AEPsych server for points to sample, but don't tell it any responses (e.g., because the participant wasn't paying attention and didn't respond). To remedy this, you can instead use `min_total_tells` to specify that a strategy will end only after a certain number of responses have been recorded (in other words, after a certain number of calls to `server.tell`). For example,  the experiment specified by the config below starts by sampling points from the Sobol sequence. After 5 responses have been recorded, it moves on to model-based optimization. The experiment ends after 100 total observations have been recorded (i.e., 5 observations from Sobol trials and 95 observations from  model-based trials).

```
[common]
parnames = [par1, par2]
lb = [0, 0]
ub = [1, 1]
outcome_type = single_probit
target = 0.75
strategy_names = [init_strat, opt_strat]

[init_strat]
generator = SobolGenerator # points will be generated from Sobol sequence
min_total_tells = 5 # next strategy will start after at least 5 responses are recorded

[opt_strat]
generator = OptimizeAcqfGenerator
model = GPClassificationModel
acqf = qNoisyExpectedImprovement
min_total_tells = 100 # experiment will end after 100 total responses are recorded
```

<h2>Ending Strategies Based on Responses</h2>
Typically AEPsych experiments start by sampling points quasi-randomly to gain some initial data to fit a model. If the participant provides the same response to all of these initialization trials, then the model may think that the participant will always provide the same response no matter what. This prevents the model from properly exploring the parameter space and makes its predictions useless. To prevent this, you can specify `min_total_outcome_occurrences` within a strategy to specify how many responses of each possible outcome are required for a strategy to finish. For example, the strategy specified in the config below will generate Sobol points until at least 10 observations have been recorded, and at least 3 of those observations need to be "yes" responses and at least 3 need to be "no" responses. The strategy will continue to generate Sobol points until ALL of these criteria have been satisfied. Note that the counts of outcome occurrences are cumulative across all strategies, so if 3 "yes"s and 2 "no"s had been collected from a previous strategy, you would only need 2 more "yes"s and 3 "no"s.

```
[example_strat]
generator = SobolGenerator # points will be generated from Sobol sequence
min_total_tells = 10 # need at least 10 recorded observations
min_total_outcome_occurrences = 3 # need at least 3 yes/no responses each
```

To prevent bad model fits when using binary outcomes, you should always have at least 1 "yes" and "no" trial each, so **the default value of `min_total_outcome_occurrences` is 1**. If you would like to disable this behavior, you must explicitly set `min_total_outcome_occurrences = 0`.

<h2>Enforcing a Hard Limit on the Duration of a Strategy</h2>
There may be scenarios where a participant really does have the same response for every point in the parameter space, so to prevent never-ending experiments, you can specify a maximum number of points a strategy should generate using using max_asks. For example, the below strategy will generate Sobol points until there are at least 10 "yes" and "no" responses each, but the strategy will end if these criteria have not been met after it has generated 50 trials.

```
[example_strat]
generator = SobolGenerator # points will be generated from Sobol sequence
min_outcome_occurrences = 10 # need at least 10 yes/no responses each
max_trials = 50 # maximum number of points to generate
```

<h2>Manually Specifying Points to be Sampled</h2>
If you want to ensure that specific points in the parameter space will be sampled, you can specify them using ManualGenerator. For example, the strategy below will sample four points in 2-d space. Note that since ManualGenerator generates a fixed number of points, other stopping criteria in the strategy will be ignored.

```
[example_strat]
generator = ManualGenerator # manually specify a set of points to sample

[ManualGenerator]
points = [[0,0], [0,1], [1,0], [1,1]] # the points to sample
shuffle = True # whether to randomize the order of the points (True by default)
```

<h2>Warm Starting a Strategy</h2>

If you want to make use of preexisting data and start your experiment with a more mature model you can do that by defining specific seed conditions. You can optionally use any data supplied under previous experiment's `[metadata]` header as filters to specify the data you'd like to use to warm start your strategy. The following is an example of how to define these seed conditions in your config: 

```
[example_strat]
generator = SobolGenerator
min_total_tells = 5
seed_data_conditions = seed_conds

[seed_conds]
experiment_name = [name_1, name_2, ... name_n]
experiment_id = [id_1, id_2, ... id_n]
experiment_description = [desc_1, desc_2, ... desc_n]
participant_id = [id_1, id_2, ... id_n]
```

<h3>Using Extra Metadata as Seed Conditions</h3>

You can also use any extra metadata as seed conditions, but rather than defining an `extra_metadata` field in your config you can define it like so: 

```
[seed_conds]
experiment_name = name_1
.
.
.
modality = [vision, haptics]
```

<h3>How is data matched?</h3>
Data is matched in three stages. In stage one, data is matched in a complex logical expression. Data in the same group are matched using a logical or operation while data is matched between groups using a logical and operation.

In stage two, the matched experiement's configurations are checked against the current experiment's configuration. Any data with a mismatch in number of expected stimuli, paramater name-type pairing, or outcome name-type paring is discarded. You will be warned about data with a continuous parameter type and mismatching bounds, however, this data will _not_ be discarded.

Finally, before the data is fed into your experiments strategy, candidate data will be transformed to ensure that no data that would be undefined in the current experiment makes it into the strategy.

<h1>Experimental Features</h1>
The following features have not yet been tested on real problems, but they are still available for use if you would like to try them.

<h2>Only Fit the Model to the Most Recent Data</h2>
AEPsych uses a model to select which points in the parameter space should be sampled next, and these choices become more intelligent as the model collects more data. This means that data collected early on in the experiment, especially during the initialization phase where points are selected quasi-randomly, may not be as informative as those collected later on. To get rid of this noise, you can specify keep_most_recent within a strategy. For example, the strategy below performs optimization for 100 trials, but on each iteration it only models the 20 most recently-collected data points.

```
[opt_strat]
generator = OptimizeAcqfGenerator
model = GPClassificationModel
acqf = qNoisyExpectedImprovement
n_asks = 100 # generate 100 points
keep_most_recent = 20 # only fit model on 20 most recent points
```

<h2>Ending Strategies Based on Model Predictions</h2>

As explained earlier, for experiments with binary outcomes we have set the default value of `min_outcome_occurrences` to 1 to prevent poor-fitting models from improperly exploring the parameter space. An alternative way to prevent this scenario is to only start model-based exploration when the model's posterior has an appropriate range of values. This can be specified using `min_post_range`. For example, the strategy below will only finish when the difference between the model posterior's minimum and maximum (in probability space) is at least 50%.

```
[init_strat]
generator = SobolGenerator
model = GPClassificationModel
min_post_range = 0.5 # required difference between min and max of the posterior probability
```

While we have yet to conduct thorough simulations to assess how well this strategy-switching rule works in practice, we are hoping that this functionality will pave the way for more sophisticated methods that will allow AEPsych to switch strategies or end data collection automatically.
