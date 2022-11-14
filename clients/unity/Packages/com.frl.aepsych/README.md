# Unity AEPsych Package

This package enables communication with an AEPsych server, and showcases several examples of using the Unity client to run experiments with the AEPsych server.

CRITICAL: This package will not work without installing and locally running the corresponding AEPsych server. Tested with AEPsych version 0.2.0.
See aepsych.org for server installation instructions.

Examples are accessed by expanding the `Samples` drop-down menu, and selecting Import on `AEPsych Client`. After import, the examples are in the scene `Assets/Samples/AEPsych Package/[version number]/AEPsych Client/Scenes/Examples.unity`. To run a given example, load the scene, select the `AEPsychExperimentSelector` object from the Hierarchy pane (default leftmost). In the Inspector pane (on the right), you
should see a panel with several sets of radio buttons. Make your selections using those buttons, and then play the scene to run the corresponding demo.

Your selection will activate the corresponding child object of the `AEPsychExperimentSelector` object. You may select that active child object to see its attached script.

This particular client communicates to the server over pysocket. To run the server corresponding to this example, from the `aepsych_server` root directory run `python aepsych_server/server.py`.

## Integrating the client into your Unity Project ##

*You will need Unity 2020.3.27.  A later version may work as well.*
To import example implementations, expand the `Samples` drop-down menu, and select Import on `AEPsych Client`.

**Add the AEPsych client to your code**

- To add an AEPsych experiment to your scene, select `AEPsych->Create New Experiment` from the toolbar. Input a unique experiment name, then select `Create`. For the rest of this explanation, replace `EXPERIMENT_NAME` with your experiment name.

**Configure your Experiment**

- To configure your experiment, select the newly added `EXPERIMENT_NAME` GameObject.
- To modify information sent to the server (experiment parameters, number of trials, etc.), edit the fields on the `Config Generator` script attatched to your `EXPERIMENT_NAME` GameObject.
- To modify the Unity interface settings, edit the fields on the `EXPERIMENT_NAME` script. Brief explanation of these settings is as follows:
- The `Response Key 0` & `Response Key 1` key codes correspond to which keys the user presses in order to respond to a trial.
- The `Use Default UI` toggle will enable/disable a basic experiment UI that prints experiment messages to the screen.
- Enabling `Use Model Exploration` instantiates a UI at runtime that enables model querying once the initialization trials are complete.
- Enabling `Auto Disable Other Canvases` disables all GameObjects in the scene that have a Canvas Component attached whenever the Model Exploration UI is toggled.
- Enabling `Record to CSV` will generate a csv file containing experiment data. See console output after exiting play mode for the generated csv's file path.
- `Start Mode` defines when the expriment will begin once playmode has been entered. `Automatic` will begin the experiment right away, `Press Any Key` will wait until any keystroke is detected, and `Manual` will wait until your own custom code calls BeginExperiment().

**Running the AEPsych Server**

For the AEPsych client to run successfully, you must have the AEPsych server downloaded and installed.
After installing, start it for connection by running the following command from the server root directory:
_python aepsych_server/server.py_

**Run your Experiment**

- To enable experiment functionality, you will need to use the parameter values suggested by the server to show some stimulus to the user. To read these values, open the `ShowStimuli` override method in the `Assets/Scripts/Experiments/EXPERIMENT_NAME` script.
- Use the dimension names that you entered into the`Experiment Dimensions` field on your experiment's `ConfigGenerator` component as keys to the `TrialConfig <string, List<float>> config` dictionary.
- Use the corresponding elements of the float list to update the state of any relevant GameObjects in your scene. See sample experiments for an example implementation.
- Start the AEPsych python server, then enter Playmode in Unity to auto-connect to that server.
