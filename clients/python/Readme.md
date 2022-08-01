# AEPsych Python client v0.1

This lets you use Python to interface with the AEPsych server to do model-based adaptive experimentation.

## Installation
We recommend installing the client under a virtual environment like
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
Once you've created a virtual environment for `AEPsychClient` and activated it, you can install through pip:

```
pip install aepsych_client
```

If you are a developer, you should also install the [main AEPsych package](https://github.com/facebookresearch/aepsych) so that you can run the tests.

## Configuration
This interface uses AEPsych's ini-based config, which gets passed as a string to the server:

```
# Instantiate a client
client = AEPsychClient(ip="0.0.0.0", port=5555)

# Send a config message to the server, passing in a configuration filename
filename = 'configs/single_lse_2d.ini'
client.configure(config_path=filename)
```

## Ask and tell
To get the next configuration from the server, we call `ask`; we report on the outcome with `tell`.

```
# Send an ask message to the server
trial_params = client.ask()

# Send a tell back
client.tell(config={"par1": [0], "par2": [1]}, outcome=1)
```

## Resume functionality
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

## Ending a session
When you are done with your experiment, you should call `client.finalize()`, which will stop the server and save your data to a database.
