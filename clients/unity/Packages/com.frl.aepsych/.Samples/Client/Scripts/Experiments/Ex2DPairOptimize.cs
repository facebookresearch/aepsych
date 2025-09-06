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

public class Ex2DPairOptimize : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex2DPairOptimize";
    #endregion

    public GameObject circlePrefab;
    GameObject circleInstance;
    GameObject exampleInstance;
    public GameObject examplePrefab;

    float startTime;

    private void Start()
    {
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
        if (exampleInstance != null)
        {
            Destroy(exampleInstance);
            exampleInstance = null;
        }
        exampleInstance = Instantiate(examplePrefab, defaultUI.transform);
        exampleInstance.transform.SetAsFirstSibling();
        exampleInstance.SetActive(true);
        SetText("Welcome. Note the color above, which is indigo.\nPress 1 to begin.");
        StartCoroutine(WaitForInput());
    }

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
        SetText("Now presenting stimulus.");

        GameObject circle1 = Instantiate(circlePrefab, new Vector3(-1.5f, 2.0f), Quaternion.identity);
        GameObject circle2 = Instantiate(circlePrefab, new Vector3(1.5f, 2.0f), Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        FlashSprite fs2 = circle2.GetComponent<FlashSprite>();
        fs.SetColor(config.GetNestedList("R")[0][0], 0.1f, config.GetNestedList("B")[0][0], 1.0f);
        fs2.SetColor(config.GetNestedList("R")[0][1], 0.1f, config.GetNestedList("B")[0][1], 1.0f);
        fs.flashDuration = 0.7f;
        fs2.flashDuration = 0.7f;

        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
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
        SetText("Press 1 or 2 for the one closer to indigo.");
        EndShowStimuli();
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

    public override void ExperimentComplete()
    {
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        fs.SetColor(maxLoc.GetFlatList("R")[0], 0.1f, maxLoc.GetFlatList("B")[0], 1.0f);
        fs.flashDuration = -1.0f; //never destroy
        SetText("Experiment complete! Displaying optimal color: ");
    }

    public override string GetName()
    {
        return experimentName;
    }

}
