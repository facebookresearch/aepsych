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

public class Ex3DPairOptimize : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex3DPairOptimize";
    #endregion

    public GameObject circlePrefab;
    public GameObject outlinePrefab;
    public GameObject examplePrefab;
    GameObject circleInstance;
    GameObject exampleInstance;
    GameObject o1;
    GameObject o2;

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
        o1 = Instantiate(outlinePrefab, new Vector3(-1.8f, 1.5f), Quaternion.identity, defaultUI.transform);
        o2 = Instantiate(outlinePrefab, new Vector3(1.8f, 1.5f), Quaternion.identity, defaultUI.transform);
        o1.GetComponentInChildren<TextMeshProUGUI>().SetText("1");
        o2.GetComponentInChildren<TextMeshProUGUI>().SetText("2");
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

        GameObject circle1 = Instantiate(circlePrefab, new Vector3(-1.8f, 1.5f), Quaternion.identity);
        GameObject circle2 = Instantiate(circlePrefab, new Vector3(1.8f, 1.5f), Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        FlashSprite fs2 = circle2.GetComponent<FlashSprite>();
        fs.SetColor(config["R"][0], config["G"][0], config["B"][0], 1.0f);
        fs2.SetColor(config["R"][1], config["G"][1], config["B"][1], 1.0f);
        fs.flashDuration = 1.4f;
        fs2.flashDuration = 1.4f;

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
        Destroy(o1);
        Destroy(o2);
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
        fs.SetColor(maxLoc["R"][0], maxLoc["G"][0], maxLoc["B"][0], 1.0f);
        fs.flashDuration = -1.0f; //never destroy
        SetText("Experiment complete! Displaying optimal color: ");
    }

    public override string GetName()
    {
        return experimentName;
    }

    public string GetCoordText(float c1, float c2)
    {
        float dist = Vector2.Distance(new Vector2(c1, c2), new Vector2(0.5f, 0.5f));
        Debug.Log("(" + System.Math.Round(c1, 3).ToString() + "," + System.Math.Round(c2, 3).ToString() + ")\ndist=" + System.Math.Round(dist, 3).ToString());
        return System.Math.Round(dist, 3).ToString();

    }
}
