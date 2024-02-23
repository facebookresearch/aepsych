# Unity client example for AEPsych Server

This project contains the Unity AEPsych package, which enables communication with an AEPsych server, and showcases several examples of using the Unity client to run experiments with the AEPsych server.

All examples are in the scene `Assets/Samples/AEPsych Package/0.0.1/AEPsych Client/Scenes/Examples.unity`. To run a given example, load the scene, select
the `AEPsychExperimentSelector` object from the Hierarchy pane (default leftmost). In the Inspector pane (on the right), you
should see a panel with several sets of radio buttons. Make your selections using those buttons, and then play the scene to run the corresponding demo.

Your selection will activate the corresponding child object of the `AEPsychExperimentSelector` object. You may select that active child object to see its attached script.

This particular client communicates to the server over pysocket. To run the server corresponding to this example, from the `aepsych_server` root directory run `python aepsych_server/server.py`.

## Integrating the client into your Unity Project ##

*You will need Unity 2022.3.20f1.  A later version may work as well.*

**Get the AEPsych Client**

- Download the Unity client
- Open Unity Hub
- Select the Project Pane
- Click "Add" in the upper-right
- Browse to the `AEPsych_unity` directory, and click "Select Folder"
- Now you can select the new `AEPsych_unity` project in Unity Hub to open it.

**Import the AEPsychClient**

- Open your project
- Select Window->Package Manager->"+" icon in top left->Add package from disk
- Browse to the `AEPsych_unity` directory that you downloaded the AEPsych client to, then select `Packages/com.frl.aepsych/package.json`.
- (Optional) To import example implementations, expand the `Samples` drop-down menu, and select Import on `AEPsych Client`.

**Add the AEPsych client to your code**

- To add an AEPsych experiment to your scene, select `AEPsych->Create New Experiment` from the toolbar. Input a unique experiment name, then select `Create`. For the rest of this explanation, replace `EXPERIMENT_NAME` with your experiment name.

**Configure your Experiment**

- To configure your experiment, select the newly added `EXPERIMENT_NAME` GameObject.
- To modify information sent to the server (experiment parameters, number of trials, etc.), edit the fields on the `Config Generator` script attatched to your `EXPERIMENT_NAME` GameObject.
- To modify the Unity interface settings, edit the fields on the `EXPERIMENT_NAME` script. Brief explanation of these settings is as follows:
- Enabling `Use Default UI` will automatically show experiment messages on screen by locating the automaticallly generated `Default UI` game object. Disabling this option in favor of a custom UI will redirect experiment messages to the console.
- Enabling `Use Model Exploration` instantiates a UI at runtime that enables model querying once the initialization trials are complete.
- Enabling `Auto Disable Other Canvases` disables all GameObjects in the scene that have a Canvas Component attached whenever the Model Exploration UI is toggled.
- Change `Start Mode` to switch experiment start-up behaviour between immediately upon entering play mode, starting after a keypress, or manual (waits for a custom script to call BeginExperiment). See the sample Experiments for an example of delaying experiment launch.
- Enabling `Record to CSV` will generate a csv file containing experiment data. See console output after exiting play mode for the generated csv's file path.

**Running the AEPsych Server**

For the AEPsych client to run successfully, you must have the AEPsych server downloaded and installed.
After installing, start it for connection by running the following command from the server root directory:
_python aepsych_server/server.py_

**Run your Experiment**

- To enable experiment functionality, you will need to use the parameter values suggested by the server to show some stimulus to the user. To read these values, open the `ShowStimuli` override method in the `Assets/Scripts/Experiments/EXPERIMENT_NAME` script.
- Use the dimension names that you entered into the`Experiment Dimensions` field on your experiment's `ConfigGenerator` component as keys to the `TrialConfig <string, List<float>> config` dictionary.
- Use the corresponding elements of the float list to update the state of any relevant GameObjects in your scene. See sample experiments for an example implementation.
- Start your AEPsych python server, then enter Playmode to auto-connect to the server.
