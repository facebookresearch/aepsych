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

public class Ex3DSingleOptimize : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex3DSingleOptimize";
    #endregion

    public GameObject circlePrefab;
    public GameObject examplePrefab;
    public TextMeshProUGUI trialText;
    public string configName = "configs/single_opt_3d.ini";

    // ShowStimuli (MANDATORY)
    // Display a stimulus, and finish when the stimulus is done.
    // Use the dimension names defined in your config file as string keys to access the float lists
    // contained in the TrialConfig. Experiment will not continue running until EndShowStimuli is called.
    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetColor(config["R"][0], config["G"][0], config["B"][0], 1.0f);
        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
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
        SetText("Was it indigo? (Y/N)");
        EndShowStimuli();
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (oldState == ExperimentState.WaitingForTellResponse)
        {
            SetText("Querying for next trial");
        }
    }

    private void Start()
    {
        GameObject example = Instantiate(examplePrefab);
        example.SetActive(true);
        SetText("Welcome. Note the color above, which is indigo. Press Y to begin.");
        StartCoroutine(WaitForInput());
    }

    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    void SetText(string s)
    {
        trialText.SetText(s);
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment complete! Displaying optimal color: ");
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    public override string GetName()
    {
        return experimentName;
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetColor(maxLoc["R"][0], maxLoc["G"][0], maxLoc["B"][0], 1.0f);
        fs.flashDuration = -1.0f; //never destroy
    }
}
