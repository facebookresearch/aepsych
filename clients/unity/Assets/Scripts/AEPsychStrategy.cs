using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AEPsych;

public class AEPsychStrategy : MonoBehaviour
{
    AEPsychClient client;
    TrialConfig baseConfig = new TrialConfig() { };
    public int stratId;
    public int totalTrials;
    public int currentTrial;
    public bool isDone = false;

    public IEnumerator InitStrat(AEPsychClient AEPsychClient, string configPath)
    {
        client = AEPsychClient;
        currentTrial = 0;
        yield return StartCoroutine(client.StartServer(configPath));
        stratId = client.GetStrat();
    }

}
