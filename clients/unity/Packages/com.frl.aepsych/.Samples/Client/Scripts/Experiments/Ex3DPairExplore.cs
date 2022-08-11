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

public class Ex3DPairExplore : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex3DPairExplore";
    #endregion

    public GameObject circlePrefab;
    public GameObject outlinePrefab;
    GameObject o1;
    GameObject o2;

    private void Start()
    {
        SetText("Connecting to server...");
    }

    public override void OnConnectToServer()
    {
        Transform canvas = defaultUI.transform;
        o1 = Instantiate(outlinePrefab, new Vector3(-1.8f, 1.5f), Quaternion.identity, canvas);
        o2 = Instantiate(outlinePrefab, new Vector3(1.8f, 1.5f), Quaternion.identity, canvas);
        o1.transform.SetAsFirstSibling();
        o2.transform.SetAsFirstSibling();
        o1.GetComponentInChildren<TextMeshProUGUI>().SetText("1");
        o2.GetComponentInChildren<TextMeshProUGUI>().SetText("2");
        SetText("Welcome. In this study, you will see two colored circles flash on the screen. You should answer which color is cooler.\n\nPress 1 to begin.");
        StartCoroutine(WaitForInput());
    }

    public IEnumerator ShowStimulusHSV(float hue1, float hue2, float sat1, float sat2, float val1, float val2)
    {
        GameObject circle1 = Instantiate(circlePrefab, new Vector3(-1.75f, 1.5f), Quaternion.identity);
        FlashSprite fs = circle1.GetComponent<FlashSprite>();
        fs.SetColor(Color.HSVToRGB(hue1, sat1, val1));
        fs.flashDuration = 1.0f;
        GameObject circle2 = Instantiate(circlePrefab, new Vector3(1.75f, 1.5f), Quaternion.identity);
        FlashSprite fs2 = circle2.GetComponent<FlashSprite>();
        fs2.SetColor(Color.HSVToRGB(hue2, sat2, val2));
        fs2.flashDuration = 1.0f;
        yield return StartCoroutine(EndShowStimuliAfterSeconds(fs2.flashDuration));
    }

    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Showing Stimuli:");
        StartCoroutine(ShowStimulusHSV(config["hue"][0], config["hue"][1], config["saturation"][0], config["saturation"][1], config["value"][0], config["value"][1]));
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
        SetText("Press 1 or 2, which color was cooler?");
        EndShowStimuli();
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        Debug.Log("State change: old: " + oldState + "new: " + newState);

        if (newState == ExperimentState.WaitingForAskResponse)
        {
            SetText("Querying for next trial");
        }

        if (queryModel.canvasGroup.alpha > 0)
        {
            DisableOutlines();
        }
        else
        {
            EnableOutlines();
        }
    }

    public override void ExperimentComplete()
    {
        base.ExperimentComplete();
        SetText("Experiment complete! Explore the model.");
    }

    void DisableOutlines()
    {
        o1.SetActive(false);
        o2.SetActive(false);
    }

    void EnableOutlines()
    {
        o1.SetActive(true);
        o2.SetActive(true);
    }

    public override string GetName()
    {
        return experimentName;
    }
}
