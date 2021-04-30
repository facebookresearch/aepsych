# AEPsych/BayesOptServer Matlab client

This lets you use matlab to interface with the AEPsych server to do model-based adaptive experimentation.

## Configuration
This interface uses AEPsych's ini-based config, which gets passed as a string to the server:

```{matlab}
%% make a config
% TODO: load this from a file
config = {'### Example config for threshold estimation from single observations. \n',
    '[common]\n',
    'lb = [1, 1]\n',
    'ub = [6, 2.9]\n',
    'outcome_type = single_probit\n',
    'parnames = [par1, par2]\n',
    'target = 0.75\n',
    '\n',
    '[experiment]\n',
    'acqf = MonotonicMCLSE\n',
    'modelbridge_cls = MonotonicSingleProbitModelbridge\n',
    'init_strat_cls = SobolStrategy\n',
    'opt_strat_cls = ModelWrapperStrategy\n',
    'model = MonotonicRejectionGP\n',
    '\n',
    '[MonotonicMCLSE]\n',
    'beta = 1.96\n',
    'objective = ProbitObjective\n',
    '\n',
    '[MonotonicRejectionGP]\n',
    'inducing_size = 100\n',
    'mean_covar_factory = default_mean_covar_factory\n',
    'monotonic_idxs = [1]\n'
    '\n',
    '[MonotonicSingleProbitModelbridge]\n',
    'restarts = 10\n',
    'samps = 1000\n',
    '\n',
    '[SobolStrategy]\n',
    'n_trials = 10\n',
    '\n',
    '[ModelWrapperStrategy]\n',
    'n_trials = 20\n',
    'refit_every = 5\n'};


%% Instantiate a client
client = AEPsychClient('port', 5555);

%% Send a config message to the server
client.configure(config);
```

## Ask and tell
To get the next configuration from the server, we call `ask`; we report on the outcome with `tell`.
```
%% Send an ask message to the server
trial_params = client.ask();

%% Send a tell back
client.tell(trial_params, 1)
```

## Resume functionality
We can run multiple interleaved experiments. When we call `configure`, we get back a strategy ID.
The client keeps track of all these strategy IDs and we can use them to resume experiments. By
doing this we can interleave different model runs.

```
%% Send another config, this generates an independent new strat
client.configure(config);

%% Run some stuff on this config
trial_params = client.ask();
client.tell(trial_params, 1)

%% Resume to a past config
client.strat_indices
client.resume(1);
trial_params = client.ask();
client.tell(trial_params, 0)
```
