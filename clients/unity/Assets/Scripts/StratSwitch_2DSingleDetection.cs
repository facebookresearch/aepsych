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


//Is this indigo?
public class StratSwitch_2DSingleDetection : MonoBehaviour
{
    public AEPsychClient client;
    TrialConfig config;
    AEPsychClient.ClientStatus clientStatus;


    bool isDone = false;
    int trialNum = 0;
    int totalTrials = 20;


    //This is specific to this example
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;
    List<AEPsychStrategy> AEPsychStrats;
    int numStrats = 2;
    int currentStrat = 0;



    //Display a stimulus, and complete when the stimulus is done
    private IEnumerator PresentStimulus(TrialConfig config)
    {
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(config["gsColor"][0], config["alpha"][0]);
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
        AEPsychStrats = new List<AEPsychStrategy>() { };
        config = new TrialConfig();

        string configPath = "Assets/StreamingAssets/configs/single_lse_2d.ini";
        for (int i = 0; i<numStrats; i++)
        {
            AEPsychStrategy b = gameObject.AddComponent<AEPsychStrategy>() as AEPsychStrategy;
            AEPsychStrats.Add(b);
            yield return StartCoroutine(b.InitStrat(client, configPath: configPath));

        }
        //start with strat 0
        yield return StartCoroutine(client.Resume(AEPsychStrats[currentStrat].stratId));

        SetText("Welcome. Press Y to begin.");
        yield return new WaitUntil(()=>Input.GetKeyDown(KeyCode.Y));
        yield return StartCoroutine(RunExperiment());
    }

    bool checkFinished()
    {
        for (int i = 0; i < numStrats; i++) {
            if (!AEPsychStrats[currentStrat].isDone)
            {
                return false;
            }
        }

        return true;
    }

    IEnumerator RunExperiment()
    {

        while (!isDone) {
            Debug.Log("trial# " + trialNum + " strat "+currentStrat + " strat's trial# " + AEPsychStrats[currentStrat].currentTrial);

            SetText("Querying for next trial");
            yield return StartCoroutine(client.Ask());

            config = client.GetConfig();

            SetText("Now presenting stimulus.");

            yield return StartCoroutine(PresentStimulus(config));

            SetText("Was it visible? (Y/N)");

            yield return StartCoroutine(LogUserInput());

            //check if strat or experiment is done
            if (AEPsychStrats[currentStrat].currentTrial == (totalTrials-1))
            {
                SetText("Strat is complete.");
                AEPsychStrats[currentStrat].isDone = true;
                if (checkFinished())
                {
                    SetText("Experiment complete");
                    isDone = true;
                    break;
                }
            }

            AEPsychStrats[currentStrat].currentTrial++;
            trialNum++;

            //in this example, switch every 5 trials
            if (AEPsychStrats[currentStrat].currentTrial % 5 == 4)
            {
                currentStrat = (currentStrat + 1) % numStrats;
                while (AEPsychStrats[currentStrat].isDone)
                {
                    currentStrat = (currentStrat + 1) % numStrats;
                }
                Debug.Log("switched to strat " + currentStrat);
                yield return StartCoroutine(client.Resume(AEPsychStrats[currentStrat].stratId));
            }




        }
        yield return 0;

    }


    void SetText(string s)
    {
        trialText.SetText(s);
    }


}
