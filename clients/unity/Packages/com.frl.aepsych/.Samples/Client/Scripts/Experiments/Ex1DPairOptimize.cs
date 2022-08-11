/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using UnityEngine;
using AEPsych;
using TMPro;

public class Ex1DPairOptimize : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex1DPairOptimize";
    #endregion

    public GameObject circlePrefab;
    GameObject circleInstance;

    float startTime;


    // ShowStimuli (MANDATORY)
    //      Modify and display a stimulus based on trial parameters.
    //      Use the dimension names defined in your config file as string
    //      keys to access the float lists contained in the TrialConfig.
    //
    // IMPORTANT: Experiment will not continue running until EndShowStimuli is called.
    //
    // EXAMPLE: to read a dimension named "dim1", write:
    //
    //      config["dim1"][0]
    //
    //      The trailing [0] indexer is necessary because the
    //      dict value is a list of floats.
    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");

        GameObject circle1 = Instantiate(circlePrefab, new Vector3(-1.2f, 2.0f), Quaternion.identity);
        GameObject circle2 = Instantiate(circlePrefab, new Vector3(1.2f, 2.0f), Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        FlashSprite fs2 = circle2.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(config["gsColor"][0]);
        fs2.SetGrayscaleColor(config["gsColor"][1]);
        fs.flashDuration = 1.0f;
        fs2.flashDuration = 1.0f;

        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
    }

    public override void CheckUserResponse()
    {
        if (startTime == -1f)
        {
            startTime = Time.time;
        }
        if (!Input.GetKeyDown(KeyCode.Alpha1) && !Input.GetKeyDown(KeyCode.Alpha2))
        {
            return;
        }
        float responseTime = Time.time - startTime;
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            ReportResultToServer(0, new TrialMetadata(responseTime, "test"));
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            ReportResultToServer(1, new TrialMetadata(responseTime, "test"));
        }
        startTime = -1f;
    }

    // BeginExperiment (optional)
    // Overrides the base BeginExperiment in order to setup the OnStateChange callback
    public override void BeginExperiment()
    {
        onStateChanged += OnExperimentStateChange;
        base.BeginExperiment();
    }

    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Alpha1));
        BeginExperiment();
    }

    // PauseExperiment (optional)
    // Overrides base PauseExperiment to remove delegate callback when paused
    public override void PauseExperiment()
    {
        onStateChanged -= OnExperimentStateChange;
        base.PauseExperiment();
    }

    // EndShowStimuliAfterSeconds (optional)
    // helper function, calls EndShowStimuli() after a number of seconds
    IEnumerator EndShowStimuliAfterSeconds(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        SetText("Press 1 or 2 for the stimulus closer to medium-gray");
        EndShowStimuli();
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (newState == ExperimentState.WaitingForAskResponse)
        {
            SetText("Querying for next trial");
        }
    }

    public override void ExperimentComplete()
    {
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        fs.flashDuration = -1.0f; //never destroy
        fs.SetGrayscaleColor(maxLoc["gsColor"][0]);
        SetText("Experiment complete! Displaying optimal color: ");
    }

    private void Start()
    {
        SetText("Connecting to server...");
    }

    public override void OnConnectToServer()
    {
        // Delete old instance when the experiment restarts
        if (circleInstance != null)
        {
            Destroy(circleInstance);
            circleInstance = null;
        }
        SetText("Welcome. Press 1 to begin.");
        StartCoroutine(WaitForInput());
    }

    public override string GetName()
    {
        return experimentName;
    }
}
