# AEPsych

AEPsych is a framework and library for adaptive experimetation in psychophysics and related domains.

## Installation
`AEPsych` only supports python 3.10+. We recommend installing `AEPsych` under a virtual environment like
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
Once you've created a virtual environment for `AEPsych` and activated it, you can install `AEPsych` using pip:

```
pip install aepsych
```

If you're a developer or want to use the latest features, you can install from GitHub using:

```
git clone https://github.com/facebookresearch/aepsych.git
cd aepsych
pip install -e .[dev]
```

## Usage
**See the code examples [here](https://github.com/facebookresearch/aepsych/tree/main/examples).**

The canonical way of using AEPsych is to launch it in server mode (you can run `aepsych_server` --help to see additional arguments):

```
aepsych_server --port 5555 --ip 0.0.0.0 --db mydatabase.db
```

The server accepts messages over a unix socket, and
all messages are formatted using [JSON](https://www.json.org/json-en.html). All messages
have the following format:

```
{
     "type":<TYPE>,
     "message":<MESSAGE>,
}
```

There are five message types: `setup`, `resume`, `ask`, `tell` and `exit` (see [aepsych/server/message_handlers](https://github.com/facebookresearch/aepsych/tree/main/aepsych/server/message_handlers) for the full set of messages).
### Setup
The `setup` message prepares the server for making suggestions and accepting data. The setup
message can be formatted as either INI or a python dict (similar to JSON) format, and an example
for psychometric threshold estimation is given in `configs/single_lse_example.ini`. It looks like this:

```
{
    "type":"setup",
    "message":{"config_str":<PASTED CONFIG STRING>}
}
```
After receiving a setup message, the server responds with a strategy index that can be used
to resume this setup (for example, for interleaving multiple experiments).

### Resume
The `resume` message tells the server to resume a strategy from earlier in the same run. It looks like this:
```
{
    "type":"resume",
    "message":{"strat_id":"0"}
}
```
After receiving a resume message, the server responds with the strategy index resumed.

### Ask
The `ask` message queries the server for the next trial configuration. It looks like this:

```
{
    "type":"ask",
    "message":""
}
```
After receiving an ask message, the server responds with a configuration in JSON format, for example
`{"frequency":100, "intensity":0.8}`

### Tell
The `tell` message updates the server with the outcome for a trial configuration. Note that the
`tell` does not need to match with a previously `ask`'d trial. For example, if you are interleaving
AEPsych runs with a classical staircase, you can still feed AEPsych with the staircase data. A message
looks like this:
```
{
    "type":"tell",
    "message":{
        "config":{
                "frequency":100,
                "intensity":0.8
            },
        "outcome":"1",
    }
}
```
### Exit
The `exit` message tells the server to close the socket connection, write strats into the database and terminate current session.
The message is:
```
{
    "type":"exit",
}
```
The server closes the connection.

## Data export and visualization
The data is logged to a SQLite database on disk (by default, `databases/default.db`). The database
has one table containing all experiment sessions that were run. Then, for each experiment there
is a table containing all messages sent and received by the server, capable of supporting a
full replay of the experiment from the server's perspective. This table can be summarized
into a data frame output (docs forthcoming) and used to visualize data (docs forthcoming).

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
AEPsych licensed CC-BY-NC 4.0, as found in the [LICENSE](LICENSE) file.

## Citing
The AEPsych paper is currently under review. In the meanwhile, you can cite our [preprint](https://arxiv.org/abs/2104.09549):

    Owen, L., Browder, J., Letham, B., Stocek, G., Tymms, C., & Shvartsman, M. (2021). Adaptive Nonparametric Psychophysics. http://arxiv.org/abs/2104.09549
