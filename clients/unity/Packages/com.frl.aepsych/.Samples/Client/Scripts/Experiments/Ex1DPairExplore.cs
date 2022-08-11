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

public class Ex1DPairExplore : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex1DPairExplore";
    #endregion

    //Specific to this example
    public GameObject sqPrefab;

    private void Start()
    {
        Transform canvas = defaultUI.transform;
        SetText("Connecting to server...");
    }

    public override void OnConnectToServer()
    {
        SetText("Welcome. In this study, you will see two shapes flash on the screen and answer which is larger. \n\nPress 1 to begin.");
        StartCoroutine(WaitForInput());
    }

    public IEnumerator ShowStimulusSizes(float size1, float size2)
    {
        GameObject sq1 = Instantiate(sqPrefab, new Vector3(-3.0f, 5.0f), Quaternion.identity);
        float sc = 0.010f;
        sq1.transform.localScale = sc * new Vector3(sq1.transform.localScale.x * size1, 1000.0f, 1.0f);

        FlashSprite fs = sq1.GetComponent<FlashSprite>();
        fs.flashDuration = 0.3f;
        yield return new WaitForSeconds(fs.flashDuration);
        yield return new WaitForSeconds(0.3f);
        GameObject sq2 = Instantiate(sqPrefab, new Vector3(3.0f, 5.0f), Quaternion.identity);

        sq2.transform.localScale = sc * new Vector3(sq2.transform.localScale.x * size2, 1000.0f, 1.0f);
        FlashSprite fs2 = sq2.GetComponent<FlashSprite>();
        fs2.flashDuration = 0.3f;
        StartCoroutine(EndShowStimuliAfterSeconds(fs2.flashDuration));
    }

    public override void ShowStimuli(TrialConfig config)
    {
        SetText("");
        StartCoroutine(ShowStimulusSizes(config["width"][0], config["width"][1]));
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
        SetText("Press 1 or 2, which was bigger?");
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
        else if (newState == ExperimentState.ConfigReady)
        {
            SetText("");
        }
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment complete!");
        base.ExperimentComplete();
    }

    public override string GetName()
    {
        return experimentName;
    }
}
