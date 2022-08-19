/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using UnityEngine;
using AEPsych;

public class ContinuousDemo : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "ContinuousDemo";
    #endregion


    public GameObject particlePrefab;
    public float fixedAlpha = 0.36f;

    public ParticleSystem particle_0;
    Vector3 spawn0 = new Vector3(0f, 4f, -2.25f);
    bool hasFinished = false;
    public float stimulusDuration = 1f;

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
        if (particle_0 == null)
        {
            particle_0 = Instantiate(particlePrefab, spawn0, Quaternion.identity).GetComponent<ParticleSystem>();
        }

        var light_0 = particle_0.lights;

        Color color = Color.HSVToRGB(config["hue"][0], 1, 1);
        color.a = fixedAlpha;

#pragma warning disable CS0618 // Type or member is obsolete

        light_0.intensityMultiplier = config["light"][0];
        particle_0.startColor = color;
        particle_0.gravityModifier = config["gravity"][0];
        particle_0.startSpeed = config["speed"][0];
        particle_0.startLifetime = config["lifetime"][0];

        // Play particle systems
        particle_0.Play();

        if (!isDone)
        {
            SetText(string.Format("To what extent does this resemble fire?"));
            EndShowStimuli();
        }
#pragma warning restore CS0618 // Type or member is obsolete
    }

    public void StopParticles()
    {
        particle_0.Stop();
        SetText("Querying Server");
    }

    // OnConnectToServer (optional)
    // Overrides the base OnConnectToServer in order to indicate when experiment is ready
    public override void OnConnectToServer()
    {
        if (startMode == StartType.PressAnyKey)
        {
            SetText("Press any key to begin.");
        }
    }

    // BeginExperiment (optional)
    // Overrides the base BeginExperiment in order to setup the OnStateChange callback
    public override void BeginExperiment()
    {
        onStateChanged += OnExperimentStateChange;
        base.BeginExperiment();
    }

    public override void Restart()
    {
        hasFinished = false;
        particle_0.Stop();
        base.Restart();
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment Complete! Optimal particle effect:");
        ShowOptimal();
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
        SetText(string.Format("Waiting for user response: [{0}]/[{1}].", ResponseKey1, ResponseKey0));
        EndShowStimuli();
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
