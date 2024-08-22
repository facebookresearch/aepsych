import time

import numpy as np
import torch

from scipy.stats import bernoulli

from scipy.special import expit, logit
from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
from aepsych.plotting import plot_strat


# Define the 75% to be where par1 + par2 = 1
def get_response_probability(params):
    m = 10
    b = logit(0.5) - m
    p = expit(m * params.sum(1) + b)
    return p


# Simulate participant responses; returns 1 if the participant detected the stimulus or 0 if they did not.
def simulate_response(trial_params):
    params = np.array([[trial_params[par][0] for par in trial_params]])
    prob = get_response_probability(params)
    response = bernoulli.rvs(prob)

    return response


# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)

# Create a server object configured to run a 2d threshold experiment
server = AEPsychServer()
client = AEPsychClient(server=server)
client.configure(
    config_path="/Users/craigsanders/fbsource/fbcode/frl/ae/aepsych/configs/single_lse_example.ini"
)

is_finished = False
while not is_finished:
    # Ask the server what the next parameter values to test should be.
    starttime = time.time()
    trial_params = client.ask()
    print(f"Ask time={time.time()-starttime}")

    # Simulate a participant response.
    outcome = simulate_response(trial_params["config"])
    # time.sleep(2)

    # Tell the server what happened so that it can update its model.
    client.tell(config=trial_params["config"], outcome=outcome)
    is_finished = trial_params["is_finished"]

# print(client.query("max"))
# Plot the results
plot_strat(server.strat, target_level=0.5)
