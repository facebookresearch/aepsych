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
        [SerializeField] bool recordToCSV;
        [Tooltip("Setting this to true will automatically record response time and send it to the server with each tell message. Timer starts on " +
            "EndShowStimuli, and continues counting through any number of trial replays.")]
        [SerializeField] bool recordResponseTime = true;
        [Tooltip("Setting Async Ask to true will query the server for the next trial while the current trial is underway, speeding up the experiment.")]
        public bool asyncAsk;

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
        [HideInInspector] public TrialConfig nextConfig;
        [HideInInspector] public AEPsychClient client;
        [HideInInspector] public AEPsychStrategy strategy;
        [HideInInspector] public CSVWriter csvFile;
        [HideInInspector] public ConfigGenerator configGenerator;
        #endregion

        // Internal Fields
        #region
        [HideInInspector] public bool isDone = false;
        [HideInInspector] public bool isPaused = false;
        [HideInInspector] public bool autoAsk = true;
        [HideInInspector] public DefaultUI defaultUI;
        [HideInInspector] public QueryModel queryModel;

        //TODO: Hide in inspector, read from server setup response
        [HideInInspector] public ResponseType responseType;
        [HideInInspector] public StimulusType stimulusType;

        string configPath;
        string prevExpText;
        bool readyToQuery = false;
        bool tellInProcess = false;
        bool hasStarted = false;
        bool nextConfigReady = false;
        float trialStartTime;
        Coroutine trialResponseListener;
        ExperimentState prevState;
        GameObject ModelExplorerPrefab;
        enum InitStatus
        {
            NotStarted,
            FirstAskSent,
            FirstAskComplete,
            SecondAskSent,
            SecondAskComplete,
            Done
        }
        InitStatus initStatus = InitStatus.NotStarted;
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
            WaitingForAsyncAsk,
            Exploring,
        };
        ExperimentState _experimentState = ExperimentState.NotConnected;

        List<ExperimentState> busyStates = new List<ExperimentState>
            {
                ExperimentState.NotConnected,
                ExperimentState.WaitingForAskResponse,
                ExperimentState.WaitingForCanModelResponse,
                ExperimentState.WaitingForQueryResponse,
                ExperimentState.WaitingForResumeResponse,
                ExperimentState.WaitingForTellResponse,
            };
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
            return;
        }

        public virtual void OnFailedToConnect()
        {
            Debug.LogError("Failed to connect. Please ensure that the server is properly installed and started.");
            SetText("Failed to connect to server.");
        }

        /// <summary>
        /// Starts experiment by beginning the first sample request
        /// </summary>
        public virtual void BeginExperiment()
        {
            AEPsychClient.Log("Starting Experiment: Strategy " + strategy.stratId);
            hasStarted = true;
            if (strategy != null && autoAsk)
                Ask(strategy);
            else if (strategy != null)
            {
                SetState(ExperimentState.WaitingForAsk);
            }
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
            StopResponseListener();
            isPaused = true;
            Disconnect();
            gameObject.SetActive(false);
        }

        public virtual void Replay()
        {
            if (IsBusy())
            {
                Debug.Log("client is busy, please wait...");
                return;
            }
            StopResponseListener();

            ShowStimuli(config);
        }

        public virtual void Restart()
        {
            if (queryModel != null && GetState() == ExperimentState.Exploring)
            {
                queryModel.HideSliders();
            }
            Disconnect();
            if (onStateChanged != null)
            {
                foreach (Delegate d in onStateChanged.GetInvocationList())
                {
                    onStateChanged -= (StateChangeCallback)d;
                }
            }
            //client.onStatusChanged -= OnStatusChanged;
            //SetState(ExperimentState.NotConnected);
            Destroy(strategy);
            strategy = null;
            isDone = false;
            isPaused = false;
            readyToQuery = false;
            hasStarted = false;
            nextConfigReady = false;
            initStatus = InitStatus.NotStarted;

            StartCoroutine(ConnectAndGenerateStrategies());
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
        /// Manually sends an ask message to the server. Call this function if you set followTellWithNewAsk equal to
        /// false in a prior call to ReportResultToServer
        /// </summary>
        public void Ask(AEPsychStrategy strat)
        {
            StartCoroutine(AskForNewConfig(strat));
        }

        /// <summary>
        /// Called when user makes a judgement and selects a response. Updates the AEPsych model with a new data point.
        /// </summary>
        public void ReportResultToServer(float outcome, TrialMetadata metadata = null, bool followTellWithNewAsk = true)
        {
            if (recordResponseTime)
            {
                float rt =  Time.time - trialStartTime;
                if (metadata == null)
                {
                    metadata = new TrialMetadata(rt);
                }
                else
                {
                    metadata.responseTime = rt;
                }
                trialStartTime = -1f;
            }
            
            if (asyncAsk && _experimentState == ExperimentState.WaitingForAsyncAsk)
            {
                AEPsychClient.Log("Received response before async ask has finished. Waiting to tell...");
                StartCoroutine(TellWhenAskReceived(outcome, metadata));
                return;
            }

            if (recordToCSV)
            {
                TrialData trial = new TrialData(DateTime.Now.ToString("hh:mm:ss"), config, outcome, metadata);
                csvFile.WriteTrial(trial);
            }
            AEPsychClient.Log(string.Format("ReportResult: {0}", outcome));
            SetState(ExperimentState.WaitingForTellResponse);
            
            StartCoroutine(client.Tell(config, outcome, metadata));

            if (useModelExploration && queryModel != null)
            {
                queryModel.QueryEnabled(false);
            }

            autoAsk = followTellWithNewAsk;

            if (asyncAsk)
            {
                config = nextConfig;
            }
        }

        IEnumerator TellWhenAskReceived(float outcome, TrialMetadata metadata = null)
        {
            yield return new WaitUntil(() => nextConfigReady);
            ReportResultToServer(outcome, metadata);
        }

        public virtual IEnumerator WaitForResponse()
        {
            yield return new WaitForSeconds(0.1f);
            yield return new WaitUntil(() => Input.GetKeyDown(ResponseKey0) || Input.GetKeyDown(ResponseKey1));
            if (Input.GetKeyDown(ResponseKey0))
            {
                ReportResultToServer(0);
            }
            else if (Input.GetKeyDown(ResponseKey1))
            {
                ReportResultToServer(1);
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
            /*
            // Resume this strat if AEPsych Client was previously handling a different one
            if (client.currentStrat != strat.stratId)
            {
                SetState(ExperimentState.WaitingForResumeResponse);
                yield return StartCoroutine(client.Resume(strat.stratId));
            }
            */
            // Don't increment trial for initial double ask
            if (asyncAsk && initStatus <= InitStatus.FirstAskComplete)
            {
                AEPsychClient.Log("Sending initial ask...");
            }
            else
            {
                strategy.currentTrial++;
                AEPsychClient.Log("strat " + strategy.stratId + " trial# "
                    + strat.currentTrial);
            }
            if (asyncAsk)
                SetState(ExperimentState.WaitingForAsyncAsk);
            else
                SetState(ExperimentState.WaitingForAskResponse);

            yield return StartCoroutine(client.Ask());
        }

        /// <summary>
        /// Called when the user gives their response for the current trial.
        /// Checks if experiment is finished and restarts the experiment loop.
        /// </summary>
        public IEnumerator EndTrial()
        {
            nextConfigReady = false;
            //check if strat or experiment is done
            if (client.finished) //check if this was the final ask for this strat
            {
                if (!isDone)
                {
                    isDone = true;
                    strategy.isDone = true;
                    ExperimentComplete();
                }
                yield break;
            }

            // Continue the experiment by asking for a new trial
            if (asyncAsk)
            {
                ShowStimuli(config);
            }
            if (autoAsk)
            {
                Ask(strategy);
            }
            else
            {
                SetState(ExperimentState.WaitingForAsk);
            }
            yield return null;
        }

        public IEnumerator WaitForAnyInput()
        {
            yield return new WaitUntil(() => Input.anyKeyDown);
            BeginExperiment();
        }

        /*
        public IEnumerator WaitForInput()
        {
            yield return new WaitUntil(() => Input.GetKeyDown(startKey));
            BeginExperiment();
        }
        */

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

            //SetState(ExperimentState.WaitingForResumeResponse);

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
            else // Initialize manually added strategy
            {
                // Ensure that the user has provided at least one strategy config
                configPath = Path.Combine(Application.streamingAssetsPath, configGenerator.GetManualConfigPath());
                if (!File.Exists(configPath))
                {
                    Debug.LogError(string.Format("Invalid manual config location: {0} To use a manual config file, place the file within Assets/StreamingAssets," +
                        " assign the file refrence in the ConfigGenerator, and ensure that the file name ends in .ini", configPath));
                    TerminateExperiment();
                    yield return null;
                }
                if (strategy == null)
                {
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

        void StopResponseListener()
        {
            if (trialResponseListener != null)
            {
                StopCoroutine(trialResponseListener);
                trialResponseListener = null;
            }
        }

        bool IsConnected()
        {
            return _experimentState != ExperimentState.NotConnected;
        }

        void Disconnect()
        {
            AEPsychClient.Log("Disconnecting " + GetName());
            StopResponseListener();
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
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                    SetState(ExperimentState.WaitingForResumeResponse);
                else if (newStatus == AEPsychClient.ClientStatus.FailedToConnect)
                {
                    OnFailedToConnect();
                }
            }
            if (_experimentState == ExperimentState.WaitingForResumeResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.Ready)
                {
                    OnConnectToServer();
                }
                else if (newStatus == AEPsychClient.ClientStatus.FailedToConnect)
                {
                    OnFailedToConnect();
                }
            }
            else if (_experimentState == ExperimentState.Exploring)
            {
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    SetState(ExperimentState.WaitingForQueryResponse);
                    if (useModelExploration)
                        queryModel.QueryEnabled(false);
                }
                else
                {
                    if (useModelExploration)
                        queryModel.QueryEnabled(true);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForTellResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    if (tellInProcess)
                    {
                        AEPsychClient.Log("Manual tell success");
                        tellInProcess = false;
                    }

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
                    if (config != null && config.Count() > 0)
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
                        if (autoAsk)
                            Ask(strategy);
                        else
                            SetState(ExperimentState.WaitingForAsk);
                    }
                }
            }
            else if (_experimentState == ExperimentState.WaitingForAskResponse)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    // We recieved a new config. Store it.
                    config = client.GetConfig();
                    SetState(ExperimentState.ConfigReady);
                    ShowStimuli(config);
                    /*
                    if (!client.finished)
                    {
                        ShowStimuli(config);
                    }
                    else
                    {
                        StartCoroutine(EndTrial());
                    }
                    */
                }
            }

            else if (_experimentState == ExperimentState.ConfigReady)
            {
                if (newStatus == AEPsychClient.ClientStatus.Ready)
                {
                    SetState(ExperimentState.WaitingForTell);
                }
            }
            else if (_experimentState == ExperimentState.WaitingForAsyncAsk)
            {
                if (newStatus == AEPsychClient.ClientStatus.GotResponse)
                {
                    // Ask twice initially, to allow async asks going forward
                    if (asyncAsk && initStatus < InitStatus.FirstAskComplete)
                    {
                        nextConfig = client.GetConfig();
                        initStatus = InitStatus.FirstAskComplete;
                        //SetState(ExperimentState.WaitingForAsyncAsk);
                        Ask(strategy);
                        return;
                    }

                    // We recieved a new config. Store it.
                    config = nextConfig;
                    nextConfig = client.GetConfig();
                    SetState(ExperimentState.ConfigReady);
                    nextConfigReady = true;
                    if (initStatus < InitStatus.SecondAskComplete)
                    {
                        ShowStimuli(config);
                        initStatus = InitStatus.Done;
                    }
                    if (useModelExploration && readyToQuery && queryModel != null)
                    {
                        queryModel.QueryEnabled(true);
                    }
                }
                else if (newStatus == AEPsychClient.ClientStatus.Ready)
                {
                    // Ask twice initially, to allow async asks going forward
                    if (asyncAsk && initStatus <= InitStatus.FirstAskComplete)
                    {
                        return;
                    }
                    else if (!asyncAsk)
                    {
                        SetState(ExperimentState.WaitingForTell);
                    }
                }
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    if (initStatus == InitStatus.NotStarted)
                    {
                        initStatus = InitStatus.FirstAskSent;
                    }
                    else if (initStatus == InitStatus.FirstAskComplete)
                    {
                        initStatus = InitStatus.SecondAskSent;
                    }
                }

            }
            else if (_experimentState == ExperimentState.WaitingForAsk)
            {
                if (newStatus == AEPsychClient.ClientStatus.QuerySent)
                {
                    if (asyncAsk)
                        SetState(ExperimentState.WaitingForAsyncAsk);
                    else
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

        /// <summary>
        /// ManualTell will send the specified config & outcome to the server. You can use this to
        /// pass domain knowledge to the AEPsych model. The manual config must match your
        /// experiment type and dimensions.
        /// </summary>
        public void ManualTell(TrialConfig manualConfig, float outcome)
        {
            if (isManualTellInProcess())
            {
                Debug.LogError("Another manual tell is already underway. Discarding the additional tell.");
                return;
            }
            StartCoroutine(ManualTellServer(manualConfig, outcome));
        }

        IEnumerator ManualTellServer(TrialConfig manualConfig, float outcome)
        {
            StopResponseListener();
            tellInProcess = true;
            yield return new WaitUntil(() => !IsBusy());
            SetState(ExperimentState.Exploring);
            if (recordToCSV)
            {
                TrialData trial = new TrialData(DateTime.Now.ToString("hh:mm:ss"), manualConfig, outcome);
                csvFile.WriteTrial(trial);
            }
            SetState(ExperimentState.WaitingForTellResponse);
            StartCoroutine(client.Tell(manualConfig, outcome));
            if (useModelExploration && queryModel != null)
            {
                queryModel.QueryEnabled(false);
            }
            if (asyncAsk)
            {
                config = nextConfig;
            }
            AEPsychClient.Log("Manual tell sent: " + manualConfig);
        }

        public bool isManualTellInProcess()
        {
            return tellInProcess;
        }

        public void RemoveConfig()
        {
            if (config != null)
            {
                config.Clear();
                config = null;
            }
        }

        public IEnumerator CheckQueryReady()
        {
            SetState(ExperimentState.WaitingForCanModelResponse);
            yield return StartCoroutine(client.CheckForModel());
        }

        /// <summary>
        /// Checks whether or not the client is currently sending/receiveing messages.
        /// </summary>
        public bool IsBusy()
        {
            if (!busyStates.Contains(_experimentState))
            {
                if (!client.IsBusy())
                    return false;
            }
            AEPsychClient.Log("Client is busy. State: " + GetState());
            return true;
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
            if (useDefaultUI)
            {
                prevExpText = defaultUI.experimentText.text;
            }
            StopResponseListener();
            SetState(ExperimentState.Exploring);
        }

        public void StopExploration()
        {
            SetState(prevState);
            if (useDefaultUI)
            {
                SetText(prevExpText);
            }
            Replay();
        }

        public void TerminateExperiment()
        {
            StopAllCoroutines();
            Disconnect();
        }

        public virtual void EndShowStimuli()
        {
            if (_experimentState == ExperimentState.Exploring)
            {
                if (queryModel != null && useModelExploration)
                {
                    queryModel.ShowSliders();
                    queryModel.QueryEnabled(true);
                }
            }
            else
            {
                if (!asyncAsk)
                {
                    SetState(ExperimentState.WaitingForTell);
                    if (useModelExploration && readyToQuery)
                    {
                        queryModel.QueryEnabled(true);
                    }
                    if (recordResponseTime)
                    {
                        if (trialStartTime == -1f)
                        {
                            trialStartTime = Time.time;
                        }
                    }
                }
                if (responseType == ResponseType.Binary)
                    trialResponseListener = StartCoroutine(WaitForResponse());
                else if (responseType == ResponseType.Continuous && useDefaultUI)
                {
                    defaultUI.ShowResponseSlider();
                }
            }
        }

        /// <summary>
        /// Resumes a paused experiment by re-enabling the Client state change listener and
        /// starting a new trial. If the experiment is uninitialized, it will initialize it.
        ///
        /// </summary>
        public void Resume()
        {
            // Skip initialization if the experiment has already started
            isPaused = false;
            if (hasStarted)
            {
                client.onStatusChanged += OnStatusChanged;
                // Resume this strat if AEPsych Client was previously handling a different one
                if (client.currentStrat != strategy.stratId)
                {
                    SetState(ExperimentState.WaitingForResumeResponse);
                    StartCoroutine(client.Resume(strategy.stratId));
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
                    Debug.Log("No Query Model game object detected in scene. Please manually place the " +
                        "ExperimentUI prefab into your scene.");
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
                else
                {
                    defaultUI.AssignActiveExperiment(this);
                }
            }
        }

        /*
        private void Update()
        {

            if (_experimentState == ExperimentState.WaitingForTell || _experimentState == ExperimentState.WaitingForAsyncAsk)
                CheckUserResponse();    // implementation of CheckUserResponse() needs to call
                                        // ReportResultToServer() when response is collected

        }
        */

        private void OnEnable()
        {
            bool waiting = false;
            // Disable any active experiments
            Experiment[] prevExperiments = FindObjectsOfType<Experiment>();
            foreach (Experiment e in prevExperiments)
            {
                if (e.GetName() != GetName() && !e.isPaused)
                {
                    AEPsychClient.Log("Pausing experiment " + e.GetName());
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
            /*
            if (queryModel != null)
            {
                queryModel.gameObject.SetActive(useModelExploration);
            }
            */
            if (!waiting)
            {
                Resume();
                AEPsychClient.Log("No experiments to pause. Resuming...");
            }
        }

        private void OnDestroy()
        {
            if (csvFile != null)
                csvFile.StopRecording();
        }

        #endregion
    }
}
