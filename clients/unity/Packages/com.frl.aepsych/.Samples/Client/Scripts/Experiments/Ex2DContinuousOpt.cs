/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using UnityEngine;
using AEPsych;

public class Ex2DContinuousOpt : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex1DContinuousOpt";
    #endregion

    public GameObject particlePrefab;
    public float stimulusDuration = 1f;
    public float fixedAlpha = 0.36f;
    public float fixedLifetime = 0.75f;
    public float fixedSpeed = 0.5f;
    Vector3 spawn = new Vector3(0f, 7f, -8f);
    float scaleFactor = 2f;
    ParticleSystem particle_0;

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
            particle_0 = Instantiate(particlePrefab, spawn, Quaternion.identity).GetComponent<ParticleSystem>();
            particle_0.transform.localScale *= scaleFactor;
        }

        Color color = Color.HSVToRGB(config.GetFlatList("hue")[0], 1, 1);
        color.a = fixedAlpha;

#pragma warning disable CS0618 // Type or member is obsolete

        particle_0.startColor = color;
        particle_0.gravityModifier = config.GetFlatList("gravity")[0];
        particle_0.startSpeed = fixedSpeed;
        particle_0.startLifetime = fixedLifetime;

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
        if (particle_0 != null)
        {
            particle_0.Stop();
            SetText("Querying Server");
        }
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

    public override void ExperimentComplete()
    {
        SetText("Experiment Complete! Optimal particle effect:");
        ShowOptimal();
    }

    public override void Restart()
    {
        particle_0.Stop();
        base.Restart();
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
            SetText("Querying for next trial...");
            StopParticles();
        }
        else if (oldState == ExperimentState.WaitingForTell)
        {
            SetText("Querying for next trial...");
            StopParticles();
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
