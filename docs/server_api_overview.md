---
id: server_overview
title: Server API Overview
---

## Server Message API
The server can handle a variety of messages from the client. The messages are
expected to be JSON objects with a `type` field that indicates the type of
message alongisde a `message` field that contains the message-specific data.
Below is an exhaustive list of the messages that the server can handle and the
expected API for each.

### setup
Sets up an experiment with the provided configuration.

**Request Format:**
```json
{
  "type": "setup",
  "message": {
    "config_str": "...",  // Config as a string, or
    "config_dict": {      // Config as a dictionary
      "common": {
        "parnames": ["param1", "param2"],
        "outcome_types": ["continuous", "binary"],
        "strategy_names": ["strategy1", "strategy2"],
      },
      "metadata": { // Metadata section is optional
        "experiment_name": "experiment_name",
        "experiment_description": "experiment_description",
        "participant_id": "participant_id"
      },
      // Additional sections for strategies, models, etc.
    }
  }
}
```

The configuration can be provided either as a string (`config_str`) or as a dictionary (`config_dict`). The configuration must include:

- A `common` section with:
  - `parnames`: List of parameter names used in the experiment
  - `outcome_types`: List of outcome types (e.g., "continuous", "binary")
  - `strategy_names`: List of stategy names.

- An optional `metadata` section with:
  - `experiment_name`: Name of the experiment
  - `experiment_description`: Description of the experiment
  - `participant_id`: ID of the participant

Additional sections should be included for configuring strategies, models, acquisition functions, and other experiment components.

**Response Format:**
```json
{
  "strat_id": 0
}
```

The `strat_id` field contains the ID of the newly created strategy.


### ask
Requests a point to be generated and returns a configuration for the next experiment trial.

**Request Format:**
```json
{
  "type": "ask",
  "message": {
    "num_points": 1  // Optional, defaults to 1
  }
}
```

The `num_points` parameter specifies how many parameter configurations to generate. If omitted, the server will generate a single configuration.

**Response Format:**
```json
{
  "config": {
    "param1": [value1],
    "param2": [value2],
    ...
  },
  "is_finished": false,
  "num_points": 1
}
```

The `config` field contains parameter values for the next trial. `is_finished` indicates whether the current strategy is finished. `num_points` indicates the number of points returned.

### tell
Tells the model which input was run and what the outcome was.

**Request Format:**
```json
{
  "type": "tell",
  "message": {
    "config": {
      "param1": value1,
      "param2": value2,
      ...
    },
    "outcome": value,  // or for multi-outcome: {"outcome1": value1, "outcome2": value2, ...}
    "model_data": true,  // Optional, defaults to true
    **kwargs, // Optional, any extra key-value pairs to add to the trial record
  }
}
```

The `config` field contains the parameter configuration used in the trial. The `outcome` field can be either a single value for single-outcome experiments or a dictionary mapping outcome names to values for multi-outcome experiments. The `model_data` parameter determines whether this data should be added to the model (true) or just recorded in the database without affecting the model (false). Any additional key-value pairs will be stored as extra data with the trial record. You can also tell the server about multiple trials at once. To do this, the parameter values and outcome values should be lists of the same length.

**Response Format:**
```json
{
  "trials_recorded": 1,
  "model_data_added": 1
}
```

The `trials_recorded` field indicates the number of trials recorded in the database. `model_data_added` indicates the number of datapoints added to the model.

### exit
Makes the server write strategies into the database and close the connection.

**Request Format:**
```json
{
  "type": "exit",
  "message": {}
}
```

This message doesn't require any parameters. It signals the server to gracefully terminate, ensuring all strategies and data are properly saved to the database before closing the connection.

**Response Format:**
```json
{
  "termination_type": "Terminate",
  "success": true
}
```

### finish_strategy
Finishes the current strategy and returns information about the finished strategy.

**Request Format:**
```json
{
  "type": "finish_strategy",
  "message": {}
}
```

This message doesn't require any parameters. It instructs the server to mark the current strategy as finished, which may trigger the server to move to the next strategy if one is available.

**Response Format:**
```json
{
  "finished_strategy": "strategy_name",
  "finished_strat_idx": 0
}
```

The `finished_strategy` field contains the name of the finished strategy. `finished_strat_idx` contains the index of the finished strategy.

### get_config
Returns the current experiment configuration.

**Request Format:**
```json
{
  "type": "get_config",
  "message": {
    "section": "section_name",  // Optional
    "property": "property_name"  // Optional, requires section to be specified
  }
}
```

The `section` parameter specifies which section of the configuration to retrieve. The `property` parameter specifies a specific property within that section to retrieve. If neither is provided, the entire configuration is returned. If `property` is specified, `section` must also be specified.

**Response Format:**
If section and property are specified:
```json
{
  "section_name": {
    "property_name": value
  }
}
```

If neither section nor property are specified, the entire config is returned:
```json
{
  "section1": {
    "property1": value1,
    "property2": value2,
    ...
  },
  "section2": {
    ...
  },
  ...
}
```

### info
Returns information about the current running experiment.

**Request Format:**
```json
{
  "type": "info",
  "message": {}
}
```

This message doesn't require any parameters. It requests comprehensive information about the current state of the experiment, including database details, strategy information, and model status.

**Response Format:**
```json
{
  "db_name": "database_name",
  "exp_id": 123,
  "strat_count": 2,
  "all_strat_names": ["strategy1", "strategy2"],
  "current_strat_index": 1,
  "current_strat_name": "strategy2",
  "current_strat_data_pts": 10,
  "current_strat_model": "model_name",
  "current_strat_acqf": "acquisition_function_name",
  "current_strat_finished": false,
  "current_strat_can_fit": true
}
```

### params
Returns information about each parameter in the current strategy.

**Request Format:**
```json
{
  "type": "parameters",
  "message": {}
}
```

This message doesn't require any parameters. It requests information about all parameters in the current strategy, including their names and bounds.

**Response Format:**
```json
{
  "param1": [lower_bound1, upper_bound1],
  "param2": [lower_bound2, upper_bound2],
  ...
}
```

Each key is a parameter name, and the value is a list containing the lower and upper bounds for that parameter.

### query
Queries the underlying model for various types of information.

**Request Format:**
```json
{
  "type": "query",
  "message": {
    "query_type": "max",  // Options: "max", "min", "prediction", "inverse"
    "probability_space": false,  // Optional, defaults to false
    "x": {  // Required for "prediction" query type
      "param1": value1,
      "param2": value2,
      ...
    },
    "y": value,  // Required for "inverse" query type
    "constraints": {  // Optional
      "0": value1,  // Parameter index: value
      "1": value2,
      ...
    }
  }
}
```

The `query_type` parameter specifies the type of query to perform:
- `max`: Find the parameter configuration that maximizes the model output
- `min`: Find the parameter configuration that minimizes the model output
- `prediction`: Get the model's prediction at a specific parameter configuration
- `inverse`: Find the parameter configuration that produces a specific output value

The `probability_space` parameter determines whether the query should be performed in probability space (true) or not (false).

The `x` parameter is required for the "prediction" query type and specifies the parameter configuration to get a prediction for.

The `y` parameter is required for the "inverse" query type and specifies the target output value to find a matching parameter configuration for.

The `constraints` parameter allows specifying equality constraints on parameters, where each key is the parameter index (0-based) and the value is the fixed value for that parameter.

**Response Format:**
```json
{
  "query_type": "max",
  "probability_space": false,
  "constraints": {
    "0": value1,
    "1": value2,
    ...
  },
  "x": {
    "param1": [value1],
    "param2": [value2],
    ...
  },
  "y": [value]
}
```

The response contains the query type, whether probability space was used, any constraints applied, the parameter configuration, and the resulting y value.

### resume
Resumes a specific strategy given its ID.

**Request Format:**
```json
{
  "type": "resume",
  "message": {
    "strat_id": 1
  }
}
```

The `strat_id` parameter specifies the ID of the strategy to resume. Note that this is not used to switch between strategies within a single experiment. The "strategy" mentioned here refers to different experiments defined by different setup messages.

**Response Format:**
```json
{
  "strat_id": 1
}
```

The `strat_id` field contains the ID of the resumed strategy.
