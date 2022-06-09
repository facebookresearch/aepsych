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

public class Ex3DSingleDetect : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex3DSingleDetect";
    #endregion

    public GameObject circlePrefab;
    GameObject circleInstance;
    public TextMeshProUGUI trialText;

    private void Start()
    {
        config = new TrialConfig();
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
        StartCoroutine(WaitForInput());
        SetText("Welcome. Press Y to begin.");
    }

    // ShowStimuli (MANDATORY)
    // Display a stimulus, and finish when the stimulus is done.
    // Use the dimension names defined in your config file as string keys to access the float lists
    // contained in the TrialConfig. Experiment will not continue running until EndShowStimuli is called.
    public override void ShowStimuli(TrialConfig config)
    {
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        Color c = Color.HSVToRGB(config["hue"][0], config["saturation"][0], 0.2f);
        fs.SetColor(c.r, c.g, c.b, config["alpha"][0]);
        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
        SetText("Now presenting stimulus.");
    }

    // BeginExperiment (optional)
    // Overrides the base BeginExperiment in order to setup the OnStateChange callback
    public override void BeginExperiment()
    {
        onStateChanged += OnExperimentStateChange;
        base.BeginExperiment();
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
        Debug.Log(string.Format("Waiting for user response: [{0}]/[{1}].", ResponseKey1, ResponseKey0));
        SetText("Was it visible? (Y/N)");
        EndShowStimuli();
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (newState == ExperimentState.WaitingForTellResponse)
        {
            SetText("Querying for next trial");
        }
    }

    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    /*
    public override void SetText(string s)
    {
        trialText.SetText(s);
    }*/

    public override void ExperimentComplete()
    {
        SetText("Experiment complete! Displaying 75% threshold color: ");
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.inverse, y: 0.75f, probability_space: true));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        fs.flashDuration = -1.0f; //never destroy
        Color c = Color.HSVToRGB(maxLoc["hue"][0], maxLoc["saturation"][0], 0.2f);
        fs.SetColor(c.r, c.g, c.b, maxLoc["alpha"][0]);
        SetText("Experiment complete! Displaying 75% threshold color: ");
    }

    public override string GetName()
    {
        return experimentName;
    }
}
