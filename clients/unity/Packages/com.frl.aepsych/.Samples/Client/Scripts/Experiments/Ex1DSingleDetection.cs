/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TMPro;
using System.Threading.Tasks;
using AEPsych;

public class Ex1DSingleDetection : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex1DSingleLse";
    #endregion

    //This is specific to this example
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;
    public string configName = "configs/single_lse_1d.ini";
    float startTime = -1f;

    // Start is called before the first frame update
    private void Start()
    {
        config = new TrialConfig();
        SetText("Welcome. Press Y to begin.");
        StartCoroutine(WaitForYInput());
    }

    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.alpha = config["alpha"][0];
        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment complete! Displaying 75% threshold color: ");
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    public override void CheckUserResponse()
    {
        if (startTime == -1f)
        {
            startTime = Time.time;
        }
        if (!Input.GetKeyDown(ResponseKey0) && !Input.GetKeyDown(ResponseKey1))
        {
            return;
        }
        float responseTime = Time.time - startTime;
        if (Input.GetKeyDown(ResponseKey0))
        {
            ReportResultToServer(0, new TrialMetadata(responseTime, "test"));
        }
        else if (Input.GetKeyDown(ResponseKey1))
        {
            ReportResultToServer(1, new TrialMetadata(responseTime, "test"));
        }
        startTime = -1f;
    }

    IEnumerator EndShowStimuliAfterSeconds(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        SetText("Was it visible? (Y/N)");
        EndShowStimuli();
    }

    // WaitForInput
    // helper function, starts study when Spacebar is pushed
    IEnumerator WaitForYInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.inverse, y: 0.75f, probability_space: true));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.alpha = maxLoc["alpha"][0];
        fs.flashDuration = -1.0f; //never destroy
    }

    new void SetText(string s)
    {
        trialText.SetText(s);
    }

    public override string GetName()
    {
        return experimentName;
    }
}
