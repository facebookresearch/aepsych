# Unity client example for BayesOpt Server 

This is an example unity client for single and pairwise bayesian optimization using the server at https://ghe.oculus-rep.com/mshvarts/bayesopt_server. 

<!--The example scenes are in `Assets/Scenes/SingleProbitGP.unity` for single/"1AFC" and `Assets/Scenes/PairwiseProbitGP.unity` for pairwise/"2AFC". The main event loops for those are in `Assets/Scripts/EventLoopSingle.cs` and `Assets/Scripts/EventLoopPairwise.cs`-->

All examples are in the scene `Assets/Scenes/Examples.unity`. To run a given example, load the scene, select 
the `BayesOptExperimentSelector` object from the Hierarchy pane (default leftmost). In the Inspector pane (on the right), you 
should see a panel with several sets of radio buttons. Make your selections using those buttons, and then play the scene to run the corresponding demo. 

Your selection will activate the corresponding child object of the `BayesOptExperimentSelector` object. You may select that active child object to see its attached script.

This particular client communicates to the server over ZMQ. To run the server corresponding to this example, install the package from the server repo and run `python bayesopt_server/server.py --socket_type zmq`. 

## Integrating the client into your Unity Project ##

*You will need Unity 2019.3.10.  A later version may work as well.*

**Get the BayesOpt Client**

- Get the Unity client from https://ghe.oculus-rep.com/mshvarts/bayesopt_unity_client  
- Click "Clone or Download"  
- Click "Download Zip"  
- Open the ZIP file.  
- Extract _bayesopt_unity_client-master_ into a temporary folder 
- Open Unity Hub  
- Select the Project Pane  
- Click "Add"  
- Browse to the _bayesopt_unity_client-master_ directory in your temporary folder, and click "Select Folder"  
- Now doubleclick the new project to open it.

**Create the BayesOpt package**
 
- Open the BayesOptClient project
- In the Project tab, expand "Scripts" and find the BayesOptClient script.  
- Right click and select "Export Package…."  
- In the "Exporting Package" window, make sure that "Include Dependencies" is checked.  
- Click "Export…"  
- Select the location to store the exported package, and give it a name like _"BayesOptClientPackage.unitypackage"_  


**Import the BayesOptClient**

- Open your project
- Select Assets->Import Package->Custom Package
- Browse to where you stored _BayesOptClientPackage.unitypackage_
- On the "Import Unity Package" window, click "all" then "Import"

**Add the BayesOpt client to your code**
- In your project, create a new, empty Game Object.  
- Name it BayesOptClient.
- In the Project pane, expand Assets->Scripts
- Drag the "BayesOptClient" script to your new BayesOptClient game object.
- In the Assets->Scripts folder are also some example script showing how to use the client for some common applications.  You can start with one of these scripts, or create your own using these as a template

**Using the BayesOpt Client**  
For the Bayesopt client to run successfully, you must have the Bayesopt server downloaded and installed. The server can be found here:  
https://ghe.oculus-rep.com/mshvarts/bayesopt_server


After installing, start it for connection by running the following command from the server root directory:  
_python bayesopt_server/server.py --socket_type zmq_

For any questions, contact @mshvarts (Michael Shvartsman). 
