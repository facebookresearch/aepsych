/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using UnityEngine;
using AEPsych;
using System.Collections.Generic;

public class ManualTellExample : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "ManualTellExample";
    #endregion

    // _________________ Manual Tell Example ________________
    // In this example, we send 10 manual tells to the server
    // when the experiment begins.
    // ______________________________________________________

    public float stimulusDuration = 1f;
    public int numManualTells = 1;
    private int trial_idx = 0;

    // IMPORTANT: To use manual tells properly, automatic Asks need to be disabled
    // on your experiment subclass. Here, when server connection is established we set
    //         autoAsk = false;
    public override void OnConnectToServer()
    {
        autoAsk = false;
        trial_idx = 0;
        SetText("Press any key to begin.");
    }

    // To send manual tells at the right time, we hook into the experiment state machine.
    // When entering the WaitingForAsk state, we will send a manual tell instead.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        // Detect the state of interest
        if (newState == ExperimentState.WaitingForAsk)
        {
            // Check if we've sent the desired number of manual tells
            if (trial_idx <= numManualTells)
            {
                if (trial_idx != numManualTells)
                {
                    // Manual tells require a new TrialConfig object. Create one.
                    TrialConfig c = new TrialConfig();
                    c.Add("Dimension 1", new List<float>() { 1 });

                    // Manual tell parameters are a TrialConfig and an outcome float.
                    ManualTell(c, 1);
                    Debug.Log("Sent a manual tell!");
                }
                // switch back to regular experiment flow when numManualTells is reached
                else
                {
                    // autoAsk needs to be true for standard state machine logic.
                    autoAsk = true;
                    // we need to trigger an ask ourselves since autoAsk was disabled.
                    Ask(strategy);
                }
                trial_idx++;
            }
        }
        AEPsychClient.Log(string.Format("SetState: {0} -> {1}", oldState, newState));
    }

    // Standard procedure after this point. No additional modifications needed.
    // ________________________________________________________________________


    public override void ShowStimuli(TrialConfig config)
    {
        SetText(string.Format("Show Stimulus: {0}", config));
        StartCoroutine(EndShowStimuliAfterSeconds(stimulusDuration));
    }

    public override void BeginExperiment()
    {
        onStateChanged += OnExperimentStateChange;
        SetText(string.Format("Sending {0} manual tells to the server...", numManualTells));
        base.BeginExperiment();
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment Complete.");
        base.ExperimentComplete();
    }

    public override void PauseExperiment()
    {
        onStateChanged -= OnExperimentStateChange;
        base.PauseExperiment();
    }

    IEnumerator EndShowStimuliAfterSeconds(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        SetText(string.Format("Waiting for user response: [{0}]/[{1}].", ResponseKey1, ResponseKey0));
        EndShowStimuli();
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
