#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aepsych import GPClassificationModel
from aepsych.config import Config

"""
Extensions scripts that can be invoked by configs to allow extending AEPsych's 
capabilities without changing AEPsych itself. These scripts will be invoked right at the
start of parsing a config. 

These extension scripts will be run by the server, thus must exist on the machine that
the extensions exist on. Extensions manipulate the server behavior, not the client. 

This example demonstrates how to add objects to the Config namespace such that they can
be used as other AEPsych objects are. 
"""


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

        # Print how many points we are using to fit.
        print(f"Fitting model on {len(train_y)} data points!")

        return super().fit(
            train_x, train_y, warmstart_hyperparams, warmstart_induc, **kwargs
        )


# You can define whatever objects you want, each of them need to be added to the config
# as below. Once this is done, they can be referred to by name in configs.
Config.register_object(VerboseGPClassificationModel)

# To invoke this extension script, the path to this script has to be added to the
# extensions option in the common block. You can have multiple textension script, each
# would be listed in extensions. In the case of registering objects to the Config, they
# can then be referred to like any other object in AEPsych, note the noisy variant
# of the model.
"""
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
"""
