# AEPsych

AEPsych is a framework and library for adaptive experimetation in psychophysics and related domains.

## Installation
`AEPsych` only supports python 3. We recommend installing `AEPsych` under a virtual environment like
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
Once you've created a virtual environment for `AEPsych` and activated it, you can install our
requirements and then install AEPsych:

```
cd aepsych
pip install -r requirements.txt
pip install -e .
```

Eventually, `AEPsych` will be available on PyPI and pip-installable directly.

## Usage
**See the code examples [here](https://github.com/facebookresearch/aepsych/tree/main/examples).**  

The canonical way of using `AEPsych` is to launch it in server mode.
```
python aepsych/server.py
```
(you can call `python aepsych/server.py --help` to see additional arguments).
The server accepts messages over either a unix socket or [ZMQ](https://zeromq.org/), and
all messages are formatted using [JSON](https://www.json.org/json-en.html). All messages
have the following format:

```
{
     "type":<TYPE>,
     "version":<VERSION>,
     "message":<MESSAGE>,
}
```
Version can be omitted, in which case we default to the oldest / unversioned handler for this message
type. There are four message types: `setup`, `resume`, `ask`, and `tell`.

### Setup
The `setup` message prepares the server for making suggestions and accepting data. The setup
message can be formatted as either INI or a python dict (similar to JSON) format, and an example
for psychometric threshold estimation is given in `configs/single_lse_example.ini`. It looks like this:

```
{
    "type":"setup",
    "version":"0.01",
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
    "version":"0.01",
    "message":{"strat_id":"0"}
}
```
After receiving a resume message, the server responds with the strategy index resumed.

### Ask
The `ask` message queries the server for the next trial configuration. It looks like this:

```
{
    "type":"ask",
    "version":"0.01",
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
    "version":"0.01",
    "message":{
        "config":{
                "frequency":100,
                "intensity":0.8
            },
        "outcome":"1",
    }
}
```

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
