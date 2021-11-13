/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AEPsych;

public class AEPsychStrategy : MonoBehaviour
{
    AEPsychClient client;
    TrialConfig baseConfig = new TrialConfig() { };
    public int stratId;
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
