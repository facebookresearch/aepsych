---
id: extensions
title: Extensions
---

This page details how to use extensions to dynamically add functionality to AEPsych 
without directly changing the package. This requires writing separate Python scripts 
that will be run when a config calls for it in a setup message.

<h2>Running Extension Scripts</h2>
In order to run extension scripts, the Config needs to know the path of the extension 
script. These can be any Python script. The extensions option is added to the common
block of a config with a list of paths to each extension to be run. The extensions are
run when the server gets the setup message with this config. Each extension script 
listed is run in the order that they are defined in common.

```ini
[common]
parnames = [signal1]
outcome_types = [binary]
stimuli_per_trial = 1
strategy_names = [opt_strat]
extensions = [./extension_script1.py, ./extension_script2.py] # Needs to be path, can be absolute, can be relative to where server is started
```

The extensions are simply Python scripts! **Do not include any extensions unless you know exactly what they do!**

Below is a common pattern to add or modify AEPsych functionality. As extensions are just
Python scripts, they can do anything a regular Python script can do and it will run on
the server.

<h3>Registering New Objects</h3>

The cleanest way to add new AEPsych functionality is to register a new object with the `Config` class.
For example, we could write a script called
`new_objects.py` containing the following code, which would add a new 
VerboseGPClassificationModel model that a config could use. This model subclasses the regular `GPClassificationModel`, so it behaves identitically except for the `fit` method, which we have overwritten to print more verbose information.

```python
from aepsych.config import Config
from aepsych import GPClassificationModel

class VerboseGPClassificationModel(GPClassificationModel):
    def fit(
        self,
        train_x,
        train_y,
        warmstart_hyperparams=False,
        warmstart_induc=False,
        **kwargs,
    ):
        """Like the usual GPClassificationModel but more verbose."""

        # Print how many points we're fitting on
        print(f"Fitting model on {len(train_y)} data points!")

        return super().fit(
            train_x, train_y, warmstart_hyperparams, warmstart_induc, **kwargs
        )


# You can define whatever objects you want, each of them need to be added to the Config
# as below. Once this is done, they can be referred to by name in configs.
Config.register_object(VerboseGPClassificationModel)
```

This new model would be used in a config as follows:

```ini
[common]
parnames = [signal1]
outcome_types = [binary]
stimuli_per_trial = 1
strategy_names = [opt_strat]
extensions = [./extensions_example/new_objects.py]

[signal1]
par_type = continuous
lower_bound = 1
upper_bound = 100

[opt_strat]
model = VerboseGPClassificationModel
generator = SobolGenerator
```

Notice how we can simply refer to the new object by its name 
"VerboseGPClassificationModel", this is available in addition to the original
GPClassificationModel as we did not modify the orignal, just subclassed it.

It is possible to unload extensions. Currently, there's no server support for unloading,
but it is available to call directly in code. You define an `_unload()` function in the
extension script that will be called when extensions are unloaded via the
ExtensionManager.

A complete script for this extenion can be found [here](https://github.com/facebookresearch/aepsych/blob/main/extensions_example/new_objects.py).
