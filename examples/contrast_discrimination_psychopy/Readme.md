# Example psychophysics experiment using psychopy

This code was used to collect the dataset in:

Letham, B., Guan, P., Tymms, C., Bakshy, E., & Shvartsman, M. (2022). Look-Ahead Acquisition Functions for Bernoulli Level Set Estimation. AISTATS 2022.

It requires psychopy in addition to the AEPsych server and client. The client isn't currently on pip and needs to be installed from github.
To install all the prerequisites:

```
pip install aepsych
pip install psychopy
git clone https://github.com/facebookresearch/aepsych.git
cd aepsych
pip install ./clients/python
```

Then, in a separate terminal window, launch the AEPsych server:

```
aepsych_server database -d contrast_sensitivity_example.db
```

Finally, launch the experiment:

```
cd examples/contrast_discrimination
python experiment.py
```
# Example psychophysics experiment using psychopy

This code was used to collect the dataset in:

Letham, B., Guan, P., Tymms, C., Bakshy, E., & Shvartsman, M. (2022). Look-Ahead Acquisition Functions for Bernoulli Level Set Estimation. AISTATS 2022.

It requires psychopy in addition to the AEPsych server and client.
To install all the prerequisites:

```
pip install aepsych
pip install aepsych_client
pip install psychopy
```

Then, in a separate terminal window, launch the AEPsych server:

```
aepsych_server database -d contrast_sensitivity_example.db
```

Finally, launch the experiment:

```
cd examples/contrast_discrimination
python experiment.py
```
