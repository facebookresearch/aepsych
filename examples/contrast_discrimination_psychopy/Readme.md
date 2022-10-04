# Example psychophysics experiment using psychopy

This code was used to collect the dataset in:

Letham, B., Guan, P., Tymms, C., Bakshy, E., & Shvartsman, M. (2022). Look-Ahead Acquisition Functions for Bernoulli Level Set Estimation. AISTATS 2022.

It requires psychopy in addition to the AEPsych server and client. Note that in practice python experiments don't have to run the server as a separate process as of AEPsych 0.2.0, but this example illustrates the full client-server interaction.
To install all the prerequisites:

```
pip install aepsych
pip install aepsych_client
pip install psychopy
```

Then, launch the AEPsych server:

```
aepsych_server database -d contrast_sensitivity_example.db
```

Finally, in a separate window, launch the experiment:

```
cd examples/contrast_discrimination
python experiment.py
```
