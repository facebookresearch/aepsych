/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using System.Threading.Tasks;
using AEPsych;

public class Ex1DSingleOpt : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex1DSingleOpt";
    #endregion

    //This is specific to this example
    public GameObject circlePrefab;
    GameObject circleInstance;
    public string configName = "configs/single_opt_1d.ini";
    float startTime = -1f;

    // Start is called before the first frame update
    private void Start()
    {
        config = new TrialConfig();
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
        StartCoroutine(WaitForInput());
        SetText("Welcome. Press Y to begin.");
    }

    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(config["gsColor"][0]);
        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
    }

    public override void ExperimentComplete()
    {
        SetText("Experiment complete! Displaying optimal color: ");
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    public override void CheckUserResponse()
    {
        if (startTime == -1f)
        {
            startTime = Time.time;
        }
        if (!Input.GetKeyDown(KeyCode.N) && !Input.GetKeyDown(KeyCode.Y))
        {
            return;
        }
        float responseTime = Time.time - startTime;
        if (Input.GetKeyDown(KeyCode.N))
        {
            ReportResultToServer(0, new TrialMetadata(responseTime, "test"));
        }
        else if (Input.GetKeyDown(KeyCode.Y))
        {
            ReportResultToServer(1, new TrialMetadata(responseTime, "test"));
        }
        startTime = -1f;
    }


    IEnumerator EndShowStimuliAfterSeconds(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        SetText("Was it medium-gray? (Y/N)");
        EndShowStimuli();
    }

    // WaitForInput
    // helper function, starts study when Spacebar is pushed
    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        circleInstance = Instantiate(circlePrefab);
        FlashSprite fs = circleInstance.GetComponent<FlashSprite>();
        fs.flashDuration = -1.0f; //never destroy
        fs.SetGrayscaleColor(maxLoc["gsColor"][0]);
    }


    public override string GetName()
    {
        return experimentName;
    }
}
