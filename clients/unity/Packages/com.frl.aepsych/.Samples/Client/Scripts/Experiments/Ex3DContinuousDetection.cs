/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using UnityEngine;
using AEPsych;
using TMPro;

public class Ex3DContinuousDetection : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex3DContinuousDetection";
    #endregion

    public float stimulusDuration = 1f;
    public GameObject circlePrefab;
    Vector2 spawnPoint = new Vector2(0f, 1.5f);

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
        SetText("Showing Stimuli:");
        StartCoroutine(ShowStimulusHSV(config.GetFlatList("hue")[0], config.GetFlatList("saturation")[0], config.GetFlatList("value")[0]));
    }

    public IEnumerator ShowStimulusHSV(float hue1, float sat1, float val1)
    {
        GameObject circle1 = Instantiate(circlePrefab, spawnPoint, Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        fs.SetColor(Color.HSVToRGB(hue1, sat1, val1));
        fs.flashDuration = 1.0f;
        yield return StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
    }

    // OnConnectToServer (optional)
    // Overrides the base OnConnectToServer in order to indicate when experiment is ready
    public override void OnConnectToServer()
    {
        defaultUI.SetResponseSliderText("cool", "warm");
        Transform canvas = defaultUI.transform;
        SetText("Welcome. In this study, you will see a colored circle flash on the screen. You should answer how warm/cool the color was.\n\nPress any key to begin.");
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
        SetText("Rate the color on a warm-cool scale.");
        EndShowStimuli();
    }

    public override void ExperimentComplete()
    {
        base.ExperimentComplete();
        SetText("Experiment complete! Explore the model.");
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (newState == ExperimentState.WaitingForAskResponse)
        {
            SetText("Querying for next trial...");
        }
        else if (oldState == ExperimentState.WaitingForTell)
        {
            SetText("Querying for next trial...");
        }
        AEPsychClient.Log(string.Format("SetState: {0} -> {1}", oldState, newState));
    }

    private void Start()
    {
        SetText("Connecting to server...");
    }
    public override string GetName()
    {
        return experimentName;
    }
}
