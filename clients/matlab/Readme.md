# AEPsych/BayesOptServer Matlab client

This lets you use matlab to interface with the AEPsych server to do model-based adaptive experimentation . The server is housed in //fbcode/frl/aepsych, or is available from facebook's internal docker hub as https://dtr.thefacebook.com/repositories/agios/bayesoptserver_master. For more info, see https://www.internalfb.com/intern/wiki/FRL/FRL_Research/FRL_Research_AEPsych/.

## Configuration
This interface uses the new ini-based config. It gets passed as a string to the server:

```{matlab}
%% make a config
% (this is ugly, we should load it from a file or have a better way of
% constructing it from a file
config_pairwise =  {'### Example config for threshold estimation from single observations. \n',
'[common]\n',
'lb = [0, 0]\n',
'ub = [1, 1]\n',
'pairwise = True\n',
'parnames = [par1, par2]\n',
'target = 0.75\n',
'\n',
'[experiment]\n',
'acqf = PairwiseMCPosteriorVariance\n',
'modelbridge_cls = PairwiseProbitBayesOpt\n',
'init_strat_cls = SobolStrategy\n',
'opt_strat_cls = ModelWrapperStrategy\n',
'model = PairwiseGP\n',
'\n',
'[PairwiseMCPosteriorVariance]\n',
'objective = ProbitObjective\n',
'\n',
'[PairwiseGP]\n',
'inducing_size = 100\n',
'mean_covar_factory = default_mean_covar_factory\n',
'\n',
'[PairwiseProbitBayesOpt]\n',
'restarts = 10\n',
'samps = 1000\n',
'\n',
'[SobolStrategy]\n',
'n_trials = 10\n',
'\n',
'[ModelWrapperStrategy]\n',
'n_trials = 20\n',
'refit_every = 5\n'};

config_single = {'### Example config for threshold estimation from single observations. \n',
    '[common]\n',
    'lb = [1, 1]\n',
    'ub = [6, 2.9]\n',
    'pairwise = False\n',
    'parnames = [distance, pressure]\n',
    'target = 0.75\n',
    '\n',
    '[experiment]\n',
    'acqf = MonotonicMCLSE\n',
    'modelbridge_cls = MonotonicSingleProbitBayesOpt\n',
    'init_strat_cls = SobolStrategy\n',
    'opt_strat_cls = ModelWrapperStrategy\n',
    'model = MonotonicGP\n',
    '\n',
    '[MonotonicMCLSE]\n',
    'beta = 1.96\n',
    'objective = ProbitObjective\n',
    '\n',
    '[MonotonicGP]\n',
    'inducing_size = 100\n',
    'mean_covar_factory = default_mean_covar_factory\n',
    'monotonic_idxs = [1]\n'
    '\n',
    '[MonotonicSingleProbitBayesOpt]\n',
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
client.configure(config_single);
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
