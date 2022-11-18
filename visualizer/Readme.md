# AEPsych Visualization Dashboard

This notebook is a visualization dashboard for AEPsych, a Python package for adaptive experimentation in psychophysics and related domains. AEPsych utilizes active learning to efficiently explore parameter spaces, allowing experimenters to find just-noticeable differences (or other quantities of interest) in far fewer trials than traditional methods. You can upload database files to this notebook to display data from earlier experiments for analysis in addition to creating new experiments. Now, you have the ability to inspect experiment configurations, raw data, and server logs. This dashboard contains built in plotting and interactive visualization that may be used for model checking.

# Installation

Requirements: anaconda, python=3.8.13

Before running the AEPsych Visualizer you will need to install AEPsych. We recommend installing the AEPsych under a virtual environment like Anaconda. You can create a virtual environment using the following commands.

```
cd aepsych
conda create -n aepsych_visualizer python=3.8.13
conda activate aepsych_visualizer
```

Once the conda environment is activated refer to the [readme.md](https://github.com/facebookresearch/aepsych#:~:text=pip%20install%20%2Dr%20requirements.txt%0Apip%20install%20%2De%20.) in the root directory to install the aepsych dependencies. After installing the dependencies you can run the notebook using the following command.

## Run the notebook with Voila

```bash
voila visualizer/AEPsych_Visualizer_Dash_Beta.ipynb
```

To stop the server click `ctrl + c`

## Deactivate virtual environment

This will remove the envroment prefix from your terminal

```bash
conda deactivate
```

## Dashboard Instructions

1) Activate the virtual environment and run the code block. You will see a set of widgets appear.

2) To resume a previous session, you can use the **"Connect to Database"** button to upload a saved .db
file and resume with all of your settings and data intact.

3) If you are starting a new experiment, you can use the "Build Experiment" button which will start the
server with a default configuration and display a dashboard with a configuration widget to change AEPsych's settings.

#### Below is an explanation of the Config Generator settings:

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

- **Initialize Model On Start:** This configures the selected model when running an initial strategy.

- **Parameters:** These settings control the parameter space that AEPsych will explore.
Use the "Add Parameter" and "Remove Parameter" buttons to add or remove parameters to the experiment. For each
parameter you can specify its name and bounds. Currently, AEPsych only supports continuous parameters.


For more information about writing config files visit aepsych.org [config docs](https://aepsych.org/docs/configs)

4) If you do not need to connect to a specific IP address and Port skip to step 5.

   Optional: You can connect to a specific IP address and Port when building a new experiment by inputing the values and clicking the **Start Server** button. Once the connection is established you will see a conformation message "Connected by Client at 127.0.0.1". Note, you must configure the server before requesting a new parameter in the interactive section of the dashboard.

5) Once you have selected your configuration setting you can click the **Submit Config File** button to configure the server.


6) You can navigate to the Interactive section of the dashboard and click the **New Parameter** button. You will see the set of parameters AEPsych recommends you try. To see a different set of parameters, click "Next Parameters" again. It may take a few seconds for the parameters to appear.

7) After testing the parameters, enter the outcome, click **Update Model** to update the model, and see the next set of recommended parameters. You can also enter data at any time with any parameter values; you are not restricted to only using the parameters that AEPsych recommends.

8) Once data is collected you can navigate to the Table View tab which displays a table containing each set of parameters and their outcome. You can download this data by clicking the **Click here to download Zip: /var/svcscm/databases/database.zip** link.

9) After the AEPsych model has been initialized, a plot of the model's posterior will appear under the plot view tab. Currently, plotting only works for 1, 2, and 3-dimensional problems.

10) To save your work, you can download the **Click here to download Zip: /var/svcscm/databases/database.zip** link at the top of the Table View tab and upload it again later.

11) If you ever need to start over, simply rerun the code block.

## Dashboard Navigation

The dashboard includes a navigation bar that allows you to interact with the AEPsych server.

**Below is an explanation of the five dashboard views:**

   - **Exp Details:** When Building a new experiment displays a configuration file to configure the AEPsych server settings. When resuming an experiment the configuration file is **view only** and is not editable as it sets the configuration based on the uploaded .db file.

   - **Plot View:** Renders a strategy plot when resuming an experiment based on the strategy from the past experiment. When building a new experiment it will initially display the message "Collect more data" until enough data is collected to plot the strategy.

   - **Table View:** Displays the parameter value, parameter name, and outcome value for all tell messages sent to the AEPsych server. This functionality is supported for new experiments and when resuming an experiment. This tab also contains a download link **/var/svcscm/databases/database.zip** at the top of the Table View tab which downloads a zipped .db file that can be saved and uploaded again later.

   When resuming an experiment the data will be replayed when the server is started. This allows you to view the plot from your past experiment.

   - **Logs:** Displays server logs that update in real-time.

   - **Interactive:** Displays a widget with three buttons "New Parameter", "Update Model", and "Send to Client"
      - **New Parameter**: When the button is clicked a UI will appear with interactive inputs.

      Below is the ask message format that is sent to the server.
      ```
      message = {
         "type": "ask",
         "version": "0.01",
         "message": ""
         }
      ```

      - **Update Model:** When the button is clicked a tell message will be sent to the server including the parameter names and values along with the outcome. A "Received msg [tell]" confirmation message will be displayed and the input and outcome values will be reset to 0.
      You can click the **New Paramete** button when you are ready for the next parameter.

      Below is the tell message format that is sent to the server.
      ```
         message = {
            "type": "tell",
            "version": "0.01",
            "message": {
                "config": {"theta": "0.19147801399230957"},
                "outcome": "1",
            },
        }
      ```

      - **Send to Client:** This functionality is still under development and the button is disabled by default.
