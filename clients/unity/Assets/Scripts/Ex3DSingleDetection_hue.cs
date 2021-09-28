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
using System.Threading.Tasks;
using AEPsych;

public class Ex3DSingleDetection_hue : MonoBehaviour
{
    //Include for all EventLoops
    public AEPsychClient client;
    TrialConfig config;

    int trialNum = 0;
    int totalTrials = 30;

    //This is specific to this example
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;


    //Display a stimulus, and complete when the stimulus is done
    private IEnumerator PresentStimulus(TrialConfig config)
    {
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        Color c = Color.HSVToRGB(config["hue"][0], config["saturation"][0], 0.2f);
        fs.SetColor(c.r, c.g, c.b, config["alpha"][0]);
        yield return new WaitForSeconds(fs.flashDuration);
    }

    //Wait for the user input; then tell the server the result
    private IEnumerator LogUserInput()
    {
        while (!Input.GetKeyDown(KeyCode.N) && !Input.GetKeyDown(KeyCode.Y))
        {
            yield return null;
        }
        if (Input.GetKeyDown(KeyCode.N))
        {
            yield return StartCoroutine(client.Tell(config, 0));
        }
        else if (Input.GetKeyDown(KeyCode.Y))
        {
            yield return StartCoroutine(client.Tell(config, 1));
        }

    }

    // Start is called before the first frame update
    IEnumerator Start()
    {
        config = new TrialConfig();
        TrialConfig baseConfig = new TrialConfig() { };
        string configPath = "Assets/StreamingAssets/configs/single_lse_3d.ini";
        yield return StartCoroutine(client.StartServer(configPath: configPath));
        SetText("Welcome. Press Y to begin.");
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        yield return StartCoroutine(RunExperiment());
    }

    IEnumerator RunExperiment()
    {
        while (!client.finished)
        {
            Debug.Log(trialNum);

            SetText("Querying for next trial");

            yield return StartCoroutine(client.Ask());

            config = client.GetConfig();

            SetText("Now presenting stimulus.");

            yield return StartCoroutine(PresentStimulus(config));

            SetText("Was it visible? (Y/N)");

            yield return StartCoroutine(LogUserInput());


            trialNum++;


        }
        SetText("Experiment complete");
        yield return 0;

    }


    void SetText(string s)
    {
        trialText.SetText(s);
    }

}
