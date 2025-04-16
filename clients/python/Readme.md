# AEPsych Python client v0.5.0

This lets you use Python to interface with the AEPsych server to do model-based adaptive experimentation.

## Installation
We recommend installing the client under a virtual environment like
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
Once you've created a virtual environment for `AEPsychClient` and activated it, you can install through pip:

```
pip install aepsych_client
```

If you are a developer, you should also install the [main AEPsych package](https://github.com/facebookresearch/aepsych) so that you can run the tests.

## Basic Usage
### Configuration
This interface uses AEPsych's ini-based config, which gets passed as a string to the server:

```
# Instantiate a client
client = AEPsychClient(ip="0.0.0.0", port=5555)

# Send a config message to the server, passing in a configuration filename
filename = 'configs/single_lse_2d.ini'
client.configure(config_path=filename)
```

### Ask and tell
To get the next configuration from the server, we call `ask`; we report on the outcome with `tell`.

```
# Send an ask message to the server
trial_params = client.ask()

# Send a tell back
client.tell(config={"par1": [0], "par2": [1]}, outcome=1)
```

### Resume functionality
We can run multiple interleaved experiments. When we call `configure`, we get back a strategy ID.
The client keeps track of all these strategy IDs and we can use them to resume experiments. By
doing this we can interleave different model runs.

```
# Configure the server using one config
client.configure(config_path=file1, config_name='config1')

# Run some stuff on this config
...

# Configure the server using another config
client.configure(config_path=file2, config_name='config2')

# Run some stuff on this other config
...

# Resume the past config
client.resume(config_name="config1)
```

### Ending a session
When you are done with your experiment, you should call `client.finalize()`, which will stop the server and save your data to a database.

## Client API

### Initialization

```python
client = AEPsychClient(ip="0.0.0.0", port=5555)
```

Creates a new AEPsych client. By default, it connects to a localhost server on port 5555.

Parameters:
- `ip` (str, optional): IP to connect to (default: "0.0.0.0")
- `port` (int, optional): Port to connect on (default: 5555)
- `connect` (bool): Connect as part of init? (default: True)
- `server` (AEPsychServer, optional): An in-memory AEPsychServer object to connect to

### Connection Management

```python
# Connect to a server
client.connect(ip="0.0.0.0", port=5555)

# End experiment and stop server
response = client.finalize()
```

The `connect` method establishes a connection to an AEPsych server at the specified IP address and port. This is automatically called during initialization if `connect=True` (the default).

Parameters:
- `ip` (str): IP address of the server to connect to
- `port` (int): Port number to connect on

The `finalize` method signals to the server that the experiment is complete and stops the server. It should be called when you're done with your experiment to ensure proper cleanup and data saving.

Returns:
- Dictionary containing confirmation of termination

### Configuration

```python
# Configure using a file path
response = client.configure(config_path="configs/my_config.ini", config_name="my_experiment")

# Configure using a config string
config_str = """
[common]
model = GPClassificationModel
acqf = MCLevelSetEstimation
...
"""
response = client.configure(config_str=config_str, config_name="my_experiment")

# Resume a previous configuration by name
client.resume(config_name="my_experiment")

# Resume a previous configuration by ID
client.resume(config_id=0)
```

The `configure` method sets up the server with an experiment configuration. You must provide either a path to a config file or a config string.

Parameters:
- `config_path` (str, optional): Path to a config.ini file
- `config_str` (str, optional): Config.ini encoded as a string
- `config_name` (str, optional): A name to assign to this config internally for convenience

Returns:
- Dictionary containing the strategy ID for the configured experiment

The `resume` method switches back to a previously configured experiment. This allows you to interleave multiple experiments in a single session.

Parameters:
- `config_id` (int, optional): ID of the config to resume
- `config_name` (str, optional): Name of the config to resume (if you provided a name during configuration)

Returns:
- Dictionary containing the strategy ID that was resumed

### Active Learning

```python
# Get next configuration to test
trial_params = client.ask()

# Get multiple configurations
trial_params = client.ask(num_points=5)

# Report results back to the server
client.tell(config={"param1": [0.5], "param2": [0.3]}, outcome=1)

# Report results with metadata
client.tell(
    config={"param1": [0.5], "param2": [0.3]},
    outcome=1,
    reaction_time=1.5,
    confidence=0.8
)

# Finish the current strategy
client.finish_strategy()
```

The `ask` method requests the next configuration(s) to test from the server. This is the core method for active learning, as it uses the model to determine the most informative points to sample next.

Parameters:
- `num_points` (int, optional): Number of points to return (default: 1)

Returns:
- Dictionary containing the suggested configuration(s) to test and whether the strategy is finished

The `tell` method reports the outcome of a tested configuration back to the server. This updates the model with new data.

Parameters:
- `config` (dict): Configuration that was evaluated (keys are parameter names, values are lists)
- `outcome` (float or dict): Outcome value(s) that were obtained
- `model_data` (bool): Whether to include this data in the model (default: True)
- `**metadata`: Additional metadata to record with this trial (e.g., reaction time, confidence)

Returns:
- Dictionary containing the number of trials recorded and data points added to the model

The `finish_strategy` method marks the current strategy as complete.

Returns:
- Dictionary containing information about the finished strategy

### Model Querying

```python
# Find the maximum of the model
max_point = client.query(query_type="max")

# Find the minimum of the model
min_point = client.query(query_type="min")

# Get model prediction at a specific point
prediction = client.query(
    query_type="prediction",
    x={"param1": [0.5], "param2": [0.3]}
)

# Get inverse prediction (what x gives a specific y)
inverse = client.query(
    query_type="inverse",
    y=0.75,
    probability_space=True
)

# Query with constraints
constrained_max = client.query(
    query_type="max",
    constraints={0: 0.5}  # Constrain first parameter to 0.5
)
```

The `query` method allows you to extract information from the underlying model. It supports several query types for different purposes.

Parameters:
- `query_type` (str): Type of query to make. Options:
  - `"max"`: Find the maximum value of the model
  - `"min"`: Find the minimum value of the model
  - `"prediction"`: Get model prediction at specific point(s)
  - `"inverse"`: Find parameter values that would give a specific outcome
- `probability_space` (bool): Whether the y value is in probability space (default: False)
- `x` (dict, optional): Parameter configuration for prediction queries
- `y` (float or tensor, optional): Expected y value for inverse queries
- `constraints` (dict, optional): Equality constraints for parameters, where keys are parameter indices and values are the fixed values
- `**kwargs`: Additional parameters to pass to the query function

Returns:
- Dictionary containing the query response, which varies based on the query type

### Information Retrieval

```python
# Get experiment information
info = client.info()

# Get parameter bounds
params = client.parameters()

# Get full experiment config
config = client.get_config()

# Get specific config property
threshold = client.get_config(section="stopping", property="min_trial_count")
```

The `info` method retrieves information about the current running experiment.

Returns:
- Dictionary containing details such as database name, experiment ID, strategy count, current strategy information, etc.

The `parameters` method gets information about each parameter in the current strategy.

Returns:
- Dictionary where keys are parameter names and values are lists containing the lower and upper bounds

The `get_config` method retrieves the current experiment configuration.

Parameters:
- `section` (str, optional): The section to get the property from
- `property` (str, optional): The property to get from the section

Returns:
- Dictionary representing the experiment config or a specific property value if section and property are provided

Each method returns a dictionary with relevant information about the operation performed. See the method docstrings for detailed information about the return values.
