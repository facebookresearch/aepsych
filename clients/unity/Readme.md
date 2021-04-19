# Unity client example for AEPsych Server

This project contains several examples of using the Unity client to run experiments with the  AEPsych server.

All examples are in the scene `Assets/Scenes/Examples.unity`. To run a given example, load the scene, select
the `AEPsychExperimentSelector` object from the Hierarchy pane (default leftmost). In the Inspector pane (on the right), you
should see a panel with several sets of radio buttons. Make your selections using those buttons, and then play the scene to run the corresponding demo.

Your selection will activate the corresponding child object of the `AEPsychExperimentSelector` object. You may select that active child object to see its attached script.

This particular client communicates to the server over ZMQ. To run the server corresponding to this example, from the `aepsych_server` root directory run `python aepsych_server/server.py --socket_type zmq`.

## Integrating the client into your Unity Project ##

*You will need Unity 2019.3.10.  A later version may work as well.*

**Get the AEPsych Client**

- Download the Unity client
- Open Unity Hub
- Select the Project Pane
- Click "Add" in the upper-right
- Browse to the `AEPsych_unity` directory, and click "Select Folder"
- Now you can select the new `AEPsych_unity` project in Unity Hub to open it.

**Create the AEPSych package**

- Open the AEPsych_unity project
- In the Project tab, expand "Scripts" and find the `AEPsychClient.cs` script.
- Right click and select "Export Package…."
- In the "Exporting Package" window, make sure that "Include Dependencies" is checked.
- Click "Export…"
- Select the location to store the exported package, and give it a name like _"AEPsychClientPackage.unitypackage"_


**Import the AEPsychClient**

- Open your project
- Select Assets->Import Package->Custom Package
- Browse to where you stored _AEPsychClientPackage.unitypackage_
- On the "Import Unity Package" window, click "all" then "Import"

**Add the AEPsych client to your code**
- In your project, create a new, empty Game Object.
- Name it AEPsychClient.
- In the Project pane, expand Assets->Scripts
- Drag the "AEPsychClient" script to your new AEPsychClient game object.
- The Assets->Scripts folder contains scripts showing how to use the client for some common applications. You can start with one of these scripts, or create your own using these as a template.
- To use AEPsych, you must load in a configuration file for your application. The StreamingAssets->configs folder contains example configuration files which should be used as templates; in most cases you only need to edit the variable names and bounds.

**Using the AEPsych Client**
For the AEPsych client to run successfully, you must have the AEPsych server downloaded and installed.

After installing, start it for connection by running the following command from the server root directory:
_python aepsych_server/server.py --socket_type zmq_
