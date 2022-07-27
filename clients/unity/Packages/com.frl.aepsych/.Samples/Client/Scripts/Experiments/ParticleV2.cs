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
using System.Collections.Generic;

public class ParticleV2 : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "ParticleV2";
    #endregion

    public GameObject particlePrefab;
    public bool isWater;
    public float fixedAlpha = 0.36f;

    public ParticleSystem particle_0;
    public ParticleSystem particle_1;
    Vector3 spawn0 = new Vector3(-2f, 4f, -2.25f);
    Vector3 spawn1 = new Vector3(2f, 4f, -2.25f);
    bool hasFinished = false;
    Vector3 mousePosition;

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
        // Spawn particle systems
        if (particle_0 == null)
        {
            particle_0 = Instantiate(particlePrefab, spawn0, Quaternion.identity).GetComponent<ParticleSystem>();
            particle_1 = Instantiate(particlePrefab, spawn1, Quaternion.identity).GetComponent<ParticleSystem>();
        }

        var light_0 = particle_0.lights;
        var light_1 = particle_1.lights;

        // If we only have one set of values, only show one particle and
        // play it indefinitely
        if (config["hue"].Count == 1)
        {
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
            hasFinished = true;
            return;
        }

        SetText(string.Format("Which looks more like {0}: Left [1] or Right [2]?", (isWater ? "water" : "fire")));

        // Convert hue into a valid color
        Color color0 = Color.HSVToRGB(config["hue"][0], 1, 1);
        color0.a = fixedAlpha;
        Color color1 = Color.HSVToRGB(config["hue"][1], 1, 1);
        color1.a = fixedAlpha;

        // Apply params to particle systems
        light_0.intensityMultiplier = config["light"][0];
        light_1.intensityMultiplier = config["light"][1];
        particle_0.startColor = color0;
        particle_1.startColor = color1;
        particle_0.gravityModifier = config["gravity"][0];
        particle_1.gravityModifier = config["gravity"][1];
        particle_0.startSpeed = config["speed"][0];
        particle_1.startSpeed = config["speed"][1];
        particle_0.startLifetime = config["lifetime"][0];
        particle_1.startLifetime = config["lifetime"][1];

        // Play particle systems
        particle_0.Play();
        particle_1.Play();

#pragma warning restore CS0618 // Type or member is obsolete
        EndShowStimuli();
        StartCoroutine(StopParticlesAfterInput());
    }

    public override void OnConnectToServer()
    {
        SetText(string.Format("Welcome. In this study, you will compare two particle effects to determine which looks more like {0}. Press any key to begin.", isWater ? "WATER" : "FIRE"));
        particle_0.transform.position = spawn0;
        particle_1.transform.position = spawn1;
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

    public override void ExperimentComplete()
    {
        SetText("Experiment Complete! Optimal particle effect:");
        ShowOptimal();
    }

    public override void OnFailedToConnect()
    {
        SetText("Failed to connect. Ensure you're connected to VPN and have added yourself to the AEPsych Permission Group.");
    }

    public override void Restart()
    {
        hasFinished = false;
        particle_0.Stop();
        particle_1.Stop();
        base.Restart();
    }

    // EndShowStimuliAfterSeconds (optional)
    // helper function, calls EndShowStimuli() after a number of seconds
    IEnumerator StopParticlesAfterInput()
    {
        yield return new WaitForSeconds(0.1f);
        yield return new WaitUntil(() => Input.GetKeyDown(ResponseKey0) || Input.GetKeyDown(ResponseKey1));
        particle_0.Stop();
        particle_1.Stop();
        SetText("Querying Server");
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (newState == ExperimentState.WaitingForTellResponse)
        {
            SetText("");
        }
        else if (newState == ExperimentState.WaitingForAskResponse)
        {
            SetText("Querying for next trial...");
        }
    }

    private void FixedUpdate()
    {
        if (hasFinished)
        {
            mousePosition = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, 6f));
            particle_0.transform.position = new Vector3(mousePosition.x, Mathf.Max(mousePosition.y, 0.5f), -2.25f);
        }
    }

    private void Start()
    {
        SetText("Connecting To Server...");
    }

    public override string GetName()
    {
        return experimentName;
    }
}
