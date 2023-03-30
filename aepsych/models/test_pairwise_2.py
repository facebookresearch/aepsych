import numpy as np

import torch

from aepsych_client import AEPsychClient

from aepsych.server import AEPsychServer

from scipy.special import expit, logit

from scipy.stats import bernoulli

import os


def get_response_probability(params):

    m = 10

    b = logit(0.75) - m

    p = expit(m * params.sum(1) + b)

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


# Configure the client/server to do pairwise optimization

server = AEPsychServer(database_path="pairwise_example.db")

client = AEPsychClient(server=server)

config_file = "../../configs/ax_pairwise_opt_example.ini"
config_file = os.path.join(os.path.dirname(__file__), config_file)

client.configure(config_file)


# Do the ask/tell loop

finished = False

while not finished:

    # Ask the server what the next parameter values to test should be.

    response = client.ask()

    trial_params = response["config"]

    finished = response["is_finished"]

    # Simulate a participant response.

    outcome = simulate_response(trial_params)

    # Tell the server what happened so that it can update its model.

    client.tell(config=trial_params, outcome=outcome)


# Finish the experiment

client.finalize()
