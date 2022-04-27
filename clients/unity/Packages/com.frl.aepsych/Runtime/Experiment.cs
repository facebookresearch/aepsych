/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TMPro;
using System.Threading.Tasks;
using UnityEngine.UI;

using AEPsych;
using UnityEditor;
using Newtonsoft.Json;
using System;
using UnityEngine.EventSystems;

namespace AEPsych
{

    public abstract class Experiment : MonoBehaviour
    {
        // Inspector Fields
        #region
        [Tooltip("Pressing this key after ShowStimuli() has completed sends a result of 0 to the server, typically representing a 'no' response.")]
        public KeyCode ResponseKey0 = KeyCode.N;
        [Tooltip("Pressing this key after ShowStimuli() has completed sends a result of 1 to the server, typically representing a 'yes' response.")]
        public KeyCode ResponseKey1 = KeyCode.Y;
        [HideInInspector, SerializeField] public KeyCode startKey = KeyCode.Space;
        public bool useDefaultUI = true;
        [SerializeField] bool useModelExploration;
        public bool autoDisableOtherCanvases;
        public bool recordToCSV;

        public enum StartType
        {
            Automatic,
            PressAnyKey,
            Manual
        }
        [SerializeField] public StartType startMode = StartType.PressAnyKey;

        #endregion

        // AEPsych Refrences
        #region
        [HideInInspector] public TrialConfig config;
        [HideInInspector] public AEPsychClient client;
        [HideInInspector] public AEPsychStrategy strategy;
        [HideInInspector] public CSVWriter csvFile;
        [HideInInspector] public ConfigGenerator configGenerator;
        #endregion

        // Internal Fields
        #region
        [HideInInspector] public bool isDone = false;
        [HideInInspector] public bool isPaused = false;
        [HideInInspector] public DefaultUI defaultUI;
        string modelExplorerPath = "Packages/com.frl.aepsych/Runtime/Prefabs/ModelExploreInterface.prefab";
        string configPath;
        bool readyToQuery = false;
        bool hasStarted = false;
        QueryModel queryModel;
        ExperimentState prevState;
        GameObject ModelExplorerPrefab;
        #endregion

        // Experiment state machine states
        #region
        public enum ExperimentState
        {
            NotConnected,
            ConfigReady,
            WaitingForResumeResponse,
            WaitingForAskResponse,
            WaitingForTellResponse,
            WaitingForQueryResponse,
            WaitingForCanModelResponse,
            WaitingForAsk,
            WaitingForTell,
            Exploring,
        };
        ExperimentState _experimentState = ExperimentState.NotConnected;
        #endregion

        // Delegate callback for changes to the experiment state machine
        public delegate void StateChangeCallback(ExperimentState oldState, ExperimentState newState);
        public event StateChangeCallback onStateChanged;


        // ______________________________ Section 1 ________________________________
        //
        //              Abstract methods for child class to implement
        // _________________________________________________________________________
        //
        public abstract void ShowStimuli(TrialConfig config);
        public abstract string GetName();


        // ______________________________ Section 2 ________________________________
        //
        //        Virtual Methods that you may want to override in child class
        // _________________________________________________________________________
        #region

        // SetText (optional)
        // Helper function to increase default visibility of trial text
        public virtual void SetText(string text)
        {
            if (useDefaultUI)
                defaultUI.SetText(text);
            else
                AEPsychClient.Log(text);
        }

        /// <summary>
        /// Called as soon as server connection is established
        /// </summary>
        public virtual void OnConnectToServer()
        {
            Debug.Log("OnConnectToServer");
            return;
        }

        /// <summary>
        /// Starts experiment by beginning the first sample request
        /// </summary>
        public virtual void BeginExperiment()
        {
            AEPsychClient.Log("Starting Experiment: Strategy " + strategy.stratId);
            hasStarted = true;
            if (strategy != null)
                StartCoroutine(AskForNewConfig(strategy));
            else
            {
                Debug.LogError("Manual config selected, but config file field is empty. " +
                    "To use manual config files, assign one in the AEPsychClient inspector " +
                    "window. To use automatic Configs, toggle \"Use Automatic Config " +
                    "Generation\" in the AEPsychClient inspector window.");
                TerminateExperiment();
            }
        }

        /// <summary>
        /// Called when all strategies are complete.
        /// </summary>
        public virtual void ExperimentComplete()
        {
            SetState(ExperimentState.Exploring);
            if (useModelExploration)
            {
                queryModel.QueryEnabled(true);
            }
            SetText("Experiment Complete");
        }

        /// <summary>
        /// Called once per frame to check for user response. Experiment will not continue running until this function calls ReportResultToServer.
        /// </summary>
        public virtual void CheckUserResponse()
        {

            if (Input.GetKeyDown(ResponseKey0))
            {
                ReportResultToServer(0);
            }
            else if (Input.GetKeyDown(ResponseKey1))
            {
                ReportResultToServer(1);
            }

        }

        public virtual void PauseExperiment()
        {
            AEPsychClient.Log("Pausing " + name);

            isPaused = true;
            Disconnect();
            gameObject.SetActive(false);
        }

        #endregion

        // ______________________________ Section 3 ________________________________
        //
        //   Default experiment methods that you most likely do not need to modify
        // _________________________________________________________________________
        #region

        public ExperimentState GetState()
        {
            return _experimentState;
        }

        void SetState(ExperimentState newState)
        {
            ExperimentState oldState = _experimentState;
            if (newState != oldState)
            {
                _experimentState = newState;
                if (onStateChanged != null)
                {
                    onStateChanged(oldState, newState);
                }
            }
        }

        /// <summary>
        /// Called when user makes a judgement and selects a response. Updates the AEPsych model with a new data point.
        /// </summary>
        public void ReportResultToServer(int outcome, TrialMetadata metadata = null)
        {
            if (recordToCSV)
            {
                TrialData trial = new TrialData(DateTime.Now.ToString("hh:mm:ss"), config, outcome, metadata);
                csvFile.WriteTrial(trial);
            }
            AEPsychClient.Log(string.Format("ReportResult: {0}", outcome));
            SetState(ExperimentState.WaitingForTellResponse);
            if (metadata != null)
                StartCoroutine(client.Tell(config, outcome, metadata));
            else
                StartCoroutine(client.Tell(config, outcome));
            if (useModelExploration)
            {
                queryModel.QueryEnabled(false);
            }
        }

        /// <summary>
        /// Called when user is ready for the next trial. Queries AEPsych server for the optimal test case.
        /// </summary>
        /// <param name="strategyId"></param>
        IEnumerator AskForNewConfig(AEPsychStrategy strat)
        {
            // Initialize strat if we haven't yet
            if (strat.stratId == -1)
            {
                yield return StartCoroutine(strat.InitStrat(client, configPath: configPath, true));
            }
            // Resume this strat if AEPsych Client was previously handling a different one
            if (client.currentStrat != strat.stratId)
            {
                SetState(ExperimentState.WaitingForResumeResponse);
                yield return StartCoroutine(client.Resume(strat.stratId));
            }
            strategy.currentTrial++;
            AEPsychClient.Log("strat " + strategy.stratId + " trial# "
                + strat.currentTrial);
            SetState(ExperimentState.WaitingForAskResponse);
            yield return StartCoroutine(client.Ask());
        }

        /// <summary>
        /// Called when the user gives their response for the current trial.
        /// Checks if experiment is finished and restarts the experiment loop.
        /// </summary>
        public IEnumerator EndTrial()
        {
            //check if strat or experiment is done
            if (isDone) //check if this was the final ask for this strat
            {
                strategy.isDone = true;
                ExperimentComplete();
                yield break;
            }

            // Continue the experiment by asking for a new trial
            StartCoroutine(AskForNewConfig(strategy));
            yield return null;
        }

        public IEnumerator WaitForAnyInput()
        {
            yield return new WaitUntil(() => Input.anyKeyDown);
            BeginExperiment();
        }

        public IEnumerator WaitForInput()
        {
            yield return new WaitUntil(() => Input.GetKeyDown(startKey));
            BeginExperiment();
        }


        /// <summary>
        /// Called when this game object becomes active. Generates strategies if they are un-initialized
        /// </summary>
        public IEnumerator ConnectAndGenerateStrategies()
        {
            yield return new WaitForSeconds(0.2f);

            // Set status change callback, which enables our study state machine logic
            client.onStatusChanged += OnStatusChanged;

            // Set up experiment params
            TrialConfig baseConfig = new TrialConfig() { };

            // Quit if experiment params are invalid
            if (!configGenerator.CheckParamValidity())
            {
                Debug.Break();
            }

            if (configGenerator.isAutomatic) // Initialize the automatically generated strategy
            {
                if (PlayerPrefs.GetString(configGenerator.experimentName + "config") == "Empty")
                {
                    Debug.LogError("Config has zero dimensions. Terminating Experiment...");
                    TerminateExperiment();
                    yield return null;
                }
                else
                {
                    strategy = gameObject.AddComponent<AEPsychStrategy>();
                    configPath = configGenerator.GetConfigText();
                    //configPath = PlayerPrefs.GetString(configGenerator.experimentName + "config");
                    yield return StartCoroutine(strategy.InitStrat(client, configPath: configPath, false));
                }
            }
            else // Initialize each manually added strategy
            {
                // Ensure that the user has provided at least one strategy config
                if (configGenerator.manualConfigFile == null)
                {
                    Debug.LogError("Must assign at least one strategy config file if using manual configs. Assign a config file in the inspector.");
                    TerminateExperiment();
                    yield return null;
                }
                if (strategy == null)
                {
                    configPath = Path.Combine(Application.dataPath, "../",
                            AssetDatabase.GetAssetPath(configGenerator.manualConfigFile));
                    strategy = gameObject.AddComponent<AEPsychStrategy>();
                    yield return StartCoroutine(strategy.InitStrat(client, configPath: configPath));
                }
            }

            // Initialize a CSV Writer for each strategy
            if (recordToCSV)
            {
                if (csvFile == null)
                    csvFile = new CSVWriter(GetName());
            }

            SetState(ExperimentState.WaitingForResumeResponse);

            // All strategies created.

            if (startMode == StartType.Automatic)
            {
                BeginExperiment();
            }
            else if (startMode == StartType.PressAnyKey)
            {
                StartCoroutine(WaitForAnyInput());
            }


        }

        bool IsConnected()
        {
            return _experimentState != ExperimentState.NotConnected;
        }

        void Disconnect()
        {
            AEPsychClient.Log("Disconnecting " + GetName());
            StopAllCoroutines();
            if (client)
            {
                client.onStatusChanged -= OnStatusChanged;
            }
            SetState(ExperimentState.NotConnected);
        }

        // Resumes this experiment when the referenced experiment
        // becomes paused
        IEnumerator ResumeWhenPaused(Experiment exp)
        {
            yield return new WaitUntil(() => exp.isPaused == true);
            Resume();
        }

        /// <summary>
        /// Callback for AEPsych Client Status Changes. This syncs the Experiment class with the AEPsychClient class and manages state transitions.
        /// // Expected order of transitions:
        /// NotConnected =>                           occurs before the experiment begins.
        ///   WaitingForResumeResponse =>             occurs after sending Resume() query to server.
        ///     WaitingForAsk =>                      occurs when resume confirmation is received.
        ///       WaitingForAskResponse =>            occurs while client awaits AEPsych server's test case selection
        ///         ConfigReady =>                    occurs when suggested test case is received from server
        ///           WaitingForTell =>               occurs while waiting for the user to respond to stimulus
        ///            WaitingForTellResponse =>     occurs as soon as user response is sent to server
        ///   WaitingForAsk => ...
        /// </summary>
        /// <param name="oldStatus"></param>
        /// <param name="newStatus"></param>
        void OnStatusChanged(AEPsychClient.ClientStatus oldStatus, AEPsychClient.ClientStatus newStatus)
        {
            AEPsychClient.Log(string.Format("OnStatusChanged Callback. Experiment status: {0}, " +
                "old client: {1}, new client: {2}", _experimentState, oldStatus, newStatus));
            if (_experimentState == ExperimentState.NotConnected)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                    OnConnectToServer();
            }
            else if (_experimentState == ExperimentState.Exploring)
            {
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    SetState(ExperimentState.WaitingForQueryResponse);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForTellResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    if (useModelExploration && !readyToQuery)
                    {
                        // Check if the model is built and ready for queries
                        StartCoroutine(CheckQueryReady());
                    }
                    else
                    {
                        StartCoroutine(EndTrial());
                    }
                }
            }
            else if (_experimentState == ExperimentState.WaitingForResumeResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    SetState(ExperimentState.WaitingForAsk);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForAskResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    // We recieved a new config. Store it.
                    config = client.GetConfig();
                    SetState(ExperimentState.ConfigReady);
                    if (!client.finished)
                    {
                        ShowStimuli(config);
                    }
                    else
                    {
                        isDone = true;
                        StartCoroutine(EndTrial());
                    }
                }
            }

            else if (_experimentState == ExperimentState.ConfigReady)
            {
                if (newStatus == AEPsychClient.ClientStatus.Ready)
                {
                    SetState(ExperimentState.WaitingForTell);
                    // Enable or Disable Model Querying based on client status
                    CheckUserResponse(); // Should call ReportResultToServer()
                                         // when response is collected
                }
            }
            else if (_experimentState == ExperimentState.WaitingForAsk)
            {
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    SetState(ExperimentState.WaitingForAskResponse);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForTell)
            {
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    SetState(ExperimentState.WaitingForTellResponse);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForQueryResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    //QueryMessage m = client.GetQueryResponse();
                    //ReceiveExplorationQuery(m.x);
                    SetState(ExperimentState.Exploring);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForCanModelResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    readyToQuery = client.GetModelResponse();
                    StartCoroutine(EndTrial());
                }
            }
        }

        #endregion

        // ______________________________ Section 4 ________________________________
        //
        //   Public methods, usually called by other scripts
        // _________________________________________________________________________

        #region
        /// <summary>
        /// ShowOptimal should be called once an experiment is complete. It will call a
        /// coroutine that queries the server for maximized probability of success,
        /// then calls ShowStimuli() with the resulting parameter values.
        /// </summary>
        public void ShowOptimal()
        {
            StartCoroutine(QueryAndDisplayOptimal());
        }

        IEnumerator QueryAndDisplayOptimal()
        {
            SetState(ExperimentState.Exploring);
            yield return StartCoroutine(client.Query(QueryType.max));
            QueryMessage m = client.GetQueryResponse();
            TrialConfig maxLoc = m.x;
            ShowStimuli(maxLoc);
        }

        public IEnumerator CheckQueryReady()
        {
            SetState(ExperimentState.WaitingForCanModelResponse);
            yield return StartCoroutine(client.CheckForModel());
        }

        public bool UsesModelExplorer()
        {
            return useModelExploration;
        }

        public void SetUsesModelExplorer(bool _useModelExploration)
        {
            useModelExploration = _useModelExploration;
        }

        public void StartExploration()
        {
            prevState = _experimentState;
            SetState(ExperimentState.Exploring);
        }

        public void StopExploration()
        {
            SetState(prevState);
        }

        public void TerminateExperiment()
        {
            StopAllCoroutines();
            Disconnect();
        }

        public void EndShowStimuli()
        {
            if (_experimentState == ExperimentState.Exploring)
            {
                if (queryModel != null)
                {
                    queryModel.ShowSliders();
                    queryModel.QueryEnabled(true);
                }
            }
            else
            {
                Debug.Log("EndShowStimuli");
                SetState(ExperimentState.WaitingForTell);
                if (useModelExploration && readyToQuery)
                {
                    queryModel.QueryEnabled(true);
                }
            }
        }

        /// <summary>
        /// Resumes a paused experiment by re-enabling the Client state change listener and
        /// starting a new trial.If the experiment is uninitialized, it will initialize it.
        ///
        /// </summary>
        public void Resume()
        {
            // Skip initialization if the experiment has already started
            isPaused = false;
            if (hasStarted)
            {
                client.onStatusChanged += OnStatusChanged;
                if (config != null)
                {
                    AEPsychClient.Log("Resuming prior trial: " + config);
                    SetState(ExperimentState.ConfigReady);
                    if (!client.finished)
                    {
                        ShowStimuli(config);
                    }
                }
                else
                {
                    StartCoroutine(AskForNewConfig(strategy));
                }
            }
            else
                StartCoroutine(ConnectAndGenerateStrategies());
        }

        #endregion

        // ______________________________ Section 5 ________________________________
        //
        //   Unity Built-in Methods
        // _________________________________________________________________________
        #region
        private void Awake()
        {
            // Find Query Model Canvas Group, if model querying is enabled in Inspector
            if (useModelExploration)
            {
                queryModel = FindObjectOfType<QueryModel>(true);
                if (queryModel == null)
                {
                    ModelExplorerPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(modelExplorerPath);
                    if (ModelExplorerPrefab != null)
                        queryModel = Instantiate(ModelExplorerPrefab).GetComponent<QueryModel>();
                    else
                        Debug.LogError(string.Format("Prefab not found at: {0}. Please Manually add the ModelExploreInterface prefab to use model exploration.", modelExplorerPath));
                    Debug.Log("No Query Model game object detected in scene. Loading default ModelExploreInterface Prefab.");
                }

                // Add Event System if there isn't one already
                if (FindObjectOfType<EventSystem>() == null)
                {
                    new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
                }
            }
            // Find client within Unity scene
            client = FindObjectOfType<AEPsychClient>();
            if (client == null)
            {
                Debug.LogError("AEPsych Client is missing. Please add the AEPsych " +
                    "Client to the scene and ensure that it is enabled.");
                TerminateExperiment();
            }
            configGenerator = GetComponent<ConfigGenerator>();
            if (configGenerator == null)
            {
                Debug.LogError("Experiement script running without a ConfigGenerator attached. " +
                    "Please attach a ConfigGenerator component to " + GetName());
                TerminateExperiment();
            }
            if (configGenerator.isAutomatic)
            {
                //AEPsychClient.Log("Automatic Config Generated:\n" + PlayerPrefs.GetString(configGenerator.experimentName + "config"));
                AEPsychClient.Log("Automatic Config Generated:\n" + configGenerator.GetConfigText());
            }
            if (useDefaultUI)
            {
                defaultUI = FindObjectOfType<DefaultUI>();
                if (defaultUI == null)
                {
                    Debug.LogError("Using default UI, but DefaultUI not found in scene. " +
                        "Please add the DefaultUI component to your scene or uncheck \"Use " +
                        "Default UI\" for your Experiment.");
                    TerminateExperiment();
                }
            }
        }

        private void Update()
        {
            if (_experimentState == ExperimentState.WaitingForTell)
                CheckUserResponse();    // implementation of CheckUserResponse() needs to call
                                        // ReportResultToServer() when response is collected
        }

        private void OnEnable()
        {
            bool waiting = false;
            // Disable any active experiments
            Experiment[] prevExperiments = FindObjectsOfType<Experiment>();
            foreach (Experiment e in prevExperiments)
            {
                if (e.GetName() != GetName())
                {
                    e.PauseExperiment();
                    StartCoroutine(ResumeWhenPaused(e));
                    waiting = true;
                }
            }

            // Enable or Disable Query Model if necessary
            if (queryModel == null)
            {
                queryModel = FindObjectOfType<QueryModel>(true);
            }
            if (queryModel != null)
            {
                queryModel.gameObject.SetActive(useModelExploration);
            }

            if (!waiting)
                Resume();
        }

        private void OnDestroy()
        {
            if (csvFile != null)
                csvFile.StopRecording();
        }

        #endregion

        // ______________________________ Section 6 ________________________________
        //
        //  Public Static Methods
        // _________________________________________________________________________
        #region
        public static GameObject LoadPrefab(string name)
        {
            return AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(AssetDatabase.FindAssets(name)[0]));
        }
        #endregion
    }
}
