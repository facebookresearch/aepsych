---
id: config_generator
title: UI Configuration Generator
---

The
[AEPsych Unity client](https://aepsych.org/docs/clients#:~:text=The%20Unity%20client%20is%20written%20in%20C%23%2C%20and%20supports%20interfacing%20AEPsych%20with%20stimulus%20display%20on%20regular%20screens%20or%20in%20VR.%20It%20additionally%20includes%20tooling%20for%20interactive%20model%20exploration%20and%20model%20querying%2C%20for%20developing%20fuller%2Dfeatured%20adaptive%20experiments%20and%20prototypes%20using%20AEPsych.)
and
[AEPsych Visualizer Dashboard](https://mybinder.org/v2/gh/Eric-Cortez/voila/main?urlpath=voila%2Frender%2Fnotebooks%2FAEPsych_Visualizer_Dash_Beta.ipynb)
allow you to configure
AEPsych without having to write any code. These clients include an interactive
UI that configures the AEPsych server.

#### Below is an explanation of the settings:

- **Initialization Trials:** Sets the number of initialization trials before the model-based strategy begins.
Parameter values during these trials are generated quasi-randomly. After the model has been initialized, it will
begin to choose parameter values according to the strategy you have picked.

- **Optimization Trials:** Sets the number of optimization trials to run.

- **Current Configuration:** The current configuration combo that was sent to the server.

- **Stimulus Type:** Sets the number of stimuli shown in each trial; 1 for single, or 2 for pairwise experiments

- **Response Type:** Sets type of response given by the participant; can be [binary] or [continuous] for single

- **Experiment Type:** There are three strategies for exploring the parameter space:

   - *Threshold Finding:* AEPsych will try to find the set of parameter values at which the outcome probability
   equals some target value.

   - *Exploration:* AEPsych will try to model the outcome at every point in the parameter space.

   - *Optimization:* AEPsych will try to find the parameter values that maximize the probability of an outcome of 1.

- **Target Threshold:** Sets the target value for the Threshold Finding strategy. It is ignored by the other strategies.

- **Initialize Model On Start:** This determines whether a model should be fit during the initialization phase of the experiment.

- **Parameters:** These settings control the parameter space that AEPsych will explore.
Use the "Add Parameter" and "Remove Parameter" buttons to add or remove parameters to the experiment. For each
parameter you can specify its name and bounds. Currently, AEPsych only supports continuous parameters.

## Valid Configuration Combinations

When selecting the stimulus, response, and experiment type the server will check
the combination of these imputs to ensure that they are valid. Currently, AEPsych
supports the following experiment combinations.

- Single Stimulus Binary Input Threshold
- Single Stimulus Binary Input Optimization
- Single Stimulus Continuous Input Threshold
- Single Stimulus Continuous Input Optimization
- Pairwise Stimulus Binary Input Optimization
- Pairwise Stimulus Binary Input Exploration

#### Example:
![Valid Combo Img](assets/config_generator.png)

The highlighted words make up a valid combination to set as the
stimulus, response, and experiment type in the GUI.

- <font color="#00BE73">Single</font> Stimulus <font color="#00BE73">Binary</font> Input <font color="#00BE73">Threshold</font>:

    - Stimulus Type: <font color="#00BE73">Single</font>
    - Response Type: <font color="#00BE73">Binary</font>
    - Experiment Type: <font color="#00BE73">Threshold</font>
