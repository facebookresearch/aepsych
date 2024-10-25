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

You cannot use the log scale transformation on parameters that include negative values
in the bounds (it is undefined). For parameters with lower bounds that include values
less than 1 (but still positive; e.g., lower bound = 0, upper bound = 2), we will
instead use `Log10(x + 1)` to transform it then use `10 ^ (x - 1)` to untransform
parameters.
