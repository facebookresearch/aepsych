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

public class Ex2DPairwiseExplore : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex2DPairwiseExplore";
    #endregion

    //Specific to this example
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;

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
        StartCoroutine(ShowStimulusHSV(config.GetNestedList("saturation")[0][0], config.GetNestedList("saturation")[0][1], config.GetNestedList("value")[0][0], config.GetNestedList("value")[0][1]));
    }
    public IEnumerator ShowStimulusHSV(float sat1, float sat2, float val1, float val2)
    {
        float hue = 0.68f;
        GameObject circle1 = Instantiate(circlePrefab, new Vector3(-3.0f, 5.0f), Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        fs.SetColor(Color.HSVToRGB(hue, sat1, val1));
        fs.flashDuration = 1.0f;
        GameObject circle2 = Instantiate(circlePrefab, new Vector3(3.0f, 5.0f), Quaternion.identity);
        FlashSprite fs2 = circle2.GetComponent<FlashSprite>();
        fs2.SetColor(Color.HSVToRGB(hue, sat2, val2));
        fs2.flashDuration = 1.0f;
        SetText("");
        yield return new WaitForSeconds(fs2.flashDuration);
        SetText("Press 1 or 2, which color was cooler?");
        EndShowStimuli();
    }

    IEnumerator Start()
    {
        SetText("Welcome. In this study, you will see two colored circles flash on the screen. You should answer which color is cooler.\n\nPress 1 to begin.");
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Alpha1));
        BeginExperiment();
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

    public override void SetText(string s)
    {
        trialText.SetText(s);
    }

    public override string GetName()
    {
        return experimentName;
    }
}
