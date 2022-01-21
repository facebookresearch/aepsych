/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using System.Threading.Tasks;
using AEPsych;

public class Ex1DSingleOpt : MonoBehaviour
{

    //Include for all EventLoops
    public AEPsychClient client;

    TrialConfig config;

    int trialNum = 0;

    //This is specific to this example
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;
    public string configName = "configs/single_opt_1d.ini";

    float reactionTime = 0.0f;


    //Display a stimulus, and complete when the stimulus is done
    private IEnumerator PresentStimulus(TrialConfig config)
    {
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(config["gsColor"][0]);
        yield return new WaitForSeconds(fs.flashDuration);
    }

    //Wait for the user input; then tell the server the result
    private IEnumerator LogUserInput()
    {
        float startTime = Time.time;
        while (!Input.GetKeyDown(KeyCode.N) && !Input.GetKeyDown(KeyCode.Y))
        {
            yield return null;
        }
        float responseTime = Time.time - startTime;
        if (Input.GetKeyDown(KeyCode.N))
        {
            yield return StartCoroutine(client.Tell(config, 0, new TrialMetadata(responseTime, "test")));
        }
        else if (Input.GetKeyDown(KeyCode.Y))
        {
            yield return StartCoroutine(client.Tell(config, 1, new TrialMetadata(responseTime, "test")));
        }

    }

    // Start is called before the first frame update
    IEnumerator Start()
    {
        config = new TrialConfig();
        string configPath = Path.Combine(Application.streamingAssetsPath, configName);
        yield return StartCoroutine(client.StartServer(configPath: configPath));
        SetText("Welcome. Press Y to begin.");
        yield return new WaitUntil(()=>Input.GetKeyDown(KeyCode.Y));
        yield return StartCoroutine(RunExperiment());
    }

    IEnumerator RunExperiment()
    {
        while (!client.finished) {
            Debug.Log(trialNum);

            SetText("Querying for next trial");

            yield return StartCoroutine(client.Ask());

            config = client.GetConfig();

            SetText("Now presenting stimulus.");

            yield return StartCoroutine(PresentStimulus(config));

            SetText("Was it medium-gray? (Y/N)");

            yield return StartCoroutine(LogUserInput());

            trialNum++;


        }

        SetText("Experiment complete! Displaying optimal color: ");
        yield return StartCoroutine(DisplayOptimal());
        yield return 0;

    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.flashDuration = -1.0f; //never destroy
        fs.SetGrayscaleColor(maxLoc["gsColor"][0]);
    }

    void SetText(string s)
    {
        trialText.SetText(s);
    }


}
