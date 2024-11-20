---
id: parameters
title: Advanced Parameter Configuration
---

This page provides an overview of additional controls for parameters, including
parameter transformations. Generally, parameters should be defined in the natural raw
parameter space and AEPsych will handle transforming the parameters into a form usable
by the models. This means that the server will always suggest parameters in response to
an `ask` in raw parameter space and you should always `tell` the server the
results of a trial also in the raw parameter space. This remains true no matter
what parameter types are used and whatever transformations are used.

<h2>Parameter types</h2>
Currently, we only support continuous parameters. More parameter types soon to come!

<h3>Continuous<h3>
```ini
[parameter]
par_type = continuous
lower_bound = 0 # any real number
upper_bound = 1 # any real number
```

Continuous parameters requires a lower bound and an upper bound to be set. Continuous
parameters can have any non-infinite ranges. This means that continuous parameters can
include negative values (e.g., lower bound = -1, upper bound = 1) or have very large
ranges (e.g., lower bound = 0, upper bound = 1,000,000).

<h3>Integer<h3>
```ini
[parameter]
par_type = integer
lower_bound = -5
upper_bound = 5
```

Integer parameters are similar to continuous parameters insofar as its possible range
and necessity of bounds. However, integer parameters will use continuous relaxation to
allow the models and generators to handle integer input/outputs. For example, this could
represent the number of lights are on for a detection threshold experiment. 

<h2>Parameter Transformations</h2>
Currently, we only support a log scale transformation to parameters. More parameter
transformations to come! In general, you can define your parameters in the raw
parameter space and AEPsych will handle the transformations for you seamlessly.

<h3>Log scale</h3>
The log scale transformation can be applied to parameters as an extra option in a
parameter-specific block.

```ini
[parameter]
par_type = continuous
lower_bound = 1 # any real number
upper_bound = 100 # any real number
log_scale = True # turn it on with any of true/yes/on, turn it off with any of false/no/off; case insensitive
```

You can log scale a parameter by adding the `log_scale` option to a parameter-specific
block and setting it to True (or true, yes, on). Log scaling is by default off. This
will transform this parameter by applying a `Log10(x)` operation to it to get to the
transformed space and apply a `10^x` operation to transformed parameters to get back to
raw parameter space.

If you use the log scale transformation on a parameter that includes values less than 1
(e.g., lower bound = -1), we will add a constant to the parameter right before we
apply the Log10 operation and subtract that constant when untransforming the parameter.
For parameters with lower bounds that are positive but still less 1, we will always use
a constant value of 1 (i.e., `Log10(x + 1)` and `10 ^ (x - 1)`). For parameters with
lower bounds that are negative, we will use a constant value of the absolute value of
the lower bound + 1 (i.e., `Log10(x + |lb| + 1)` and `10 ^ (x - |lb| - 1)`).

<h3>Normalize scale</h3>
By default, all parameters will have their scale min-max normalized to the range of 
[0, 1]. This prevents any particular parameter with a large scale to completely dominate
the other parameters. Very rarely, this behavior may not be desired and can be turned 
off for specific parameters.

```ini
[parameter]
par_type = continuous
lower_bound = 1 
upper_bound = 100
normalize_scale = False # turn it on with any of true/yes/on, turn it off with any of false/no/off; case insensitive
```

By setting the `normalize_scale` option to False, this parameter will not be scaled 
before being given to the model and therefore maintain its original magnitude. This is
very rarely necessary and should be used with caution. 

<h2>Order of operations</h2>
Parameter types and parameter-specific transforms are all handled by the 
`ParameterTransform` API. Transforms built from config files will have a specific order
of operation, regardless of how the options were set in the config file. Each parameter
is transformed entirely separately. 

Currently, the order is as follows:
* Rounding for integer parameters (rounding is applied in both directions)
* Log scale
* Normalize scale