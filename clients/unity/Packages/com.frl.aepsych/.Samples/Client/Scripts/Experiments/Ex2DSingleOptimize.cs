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

public class Ex2DSingleOptimize : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex2DSingleOptimize";
    #endregion

    //This is specific to this example
    public GameObject circlePrefab;
    public GameObject examplePrefab;
    public TextMeshProUGUI trialText;
    public string configName = "configs/single_opt_2d.ini";
    public Canvas experimentCanvas;

    // ShowStimuli (MANDATORY)
    // Display a stimulus, and finish when the stimulus is done.
    // Use the dimension names defined in your config file as string keys to access the float lists
    // contained in the TrialConfig. Experiment will not continue running until EndShowStimuli is called.
    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetColor(config["R"][0], 0.2f, config["B"][0], 1.0f);
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
        example.transform.parent = experimentCanvas.transform;
        example.SetActive(true);
        SetText("Welcome. Note the color above, which is indigo. Press Y to begin.");
        StartCoroutine(WaitForInput());
    }

    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    public override void SetText(string s)
    {
        trialText.SetText(s);
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment complete! Displaying optimal color: ");
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.flashDuration = -1.0f; //never destroy
        fs.SetColor(maxLoc["R"][0], 0.2f, maxLoc["B"][0], 1.0f);
    }

    public override string GetName()
    {
        return experimentName;
    }
}
