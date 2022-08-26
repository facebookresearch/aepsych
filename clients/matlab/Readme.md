## AEPsych/BayesOptServer Matlab client

This lets you use matlab to interface with the AEPsych server to do model-based adaptive experimentation.

## Configuration
This interface uses AEPsych's ini-based config, which gets passed as a string to the server:

```{matlab}

%% Instantiate a client
client = AEPsychClient('port', 5555);

%% Send a config message to the server, passing in a configuration filename
filename = 'configs/single_lse_example.ini';
client.configure_by_file(filename);
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
client.configure_by_file(filename);

%% Run some stuff on this config
trial_params = client.ask();
client.tell(trial_params, 1)

%% Resume to a past config
client.strat_indices
client.resume(1);
trial_params = client.ask();
client.tell(trial_params, 0)
```


## Querying functionality
We can query the models to get the max location, the value at a particular
location, or the location of a target value. For single models, you can
also set the message to use probability space. Note that for querying to work,
the current strategy in the server must have a model, meaning that it
must be past the initialization trials.

```

%% Check if the strategy has a model
client.get_can_model()

%% Get max value and its position
[val, loc] = client.get_max()

%% Query the client at that position
client.predict(loc, false)

%% Inverse query to find the value
client.find_val(val, false)
```
