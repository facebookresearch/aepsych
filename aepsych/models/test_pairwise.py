import numpy as np

import torch
import os


from aepsych.server import AEPsychServer

from aepsych_client import AEPsychClient

from ax.service.utils.report_utils import exp_to_df

from scipy.special import expit, logit

from scipy.stats import bernoulli


# Define the 75% to be where par1 + par2 = 1


def get_response_probability(params):

    m = 10

    b = logit(0.75) - m

    par1 = float(params[0][0])

    par2 = float(params[0][1])

    p = expit(m * (par1 + par2) + b)

    return p


# Simulate participant responses; returns 1 if the participant detected the stimulus or 0 if they did not.


def simulate_response(trial_params):

    params = np.array([[trial_params[par][0] for par in trial_params]])
    p = get_response_probability(params)

    response = bernoulli.rvs(p)

    return response


# Fix random seeds

np.random.seed(0)

torch.manual_seed(0)


# Create a server object configured to run a 2d threshold experiment

client = AEPsychClient(server=AEPsychServer())

config_file = "../../configs/ax_example.ini"
config_file = os.path.join(os.path.dirname(__file__), config_file)
client.configure(config_file)


while not client.server.strat.finished:

    # Ask the server what the next parameter values to test should be.

    trial_params = client.ask()

    # Simulate a participant response.

    outcome = simulate_response(trial_params["config"])

    # Tell the server what happened so that it can update its model.

    client.tell(trial_params["config"], outcome)

    print(exp_to_df(client.server.strat.ax_client.experiment))
