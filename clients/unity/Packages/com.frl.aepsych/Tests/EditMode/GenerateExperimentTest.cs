/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using AEPsych;
using UnityEditor;

public class GenerateExperimentTest
{
    string testExperimentName = "UnitTestExperiment";
    ExperimentGenerator generator;

    [UnityTest]
    public IEnumerator A_InitExperimentGeneration()
    {
        Debug.unityLogger.logEnabled = true;
        generator = (ExperimentGenerator)ExperimentGenerator.CreateNew();
        generator.experimentName = testExperimentName;
        generator.CreateNewExperiment();
        Assert.IsTrue(true);
        yield return null;
    }

    [UnityTest]
    public IEnumerator B_WaitForCompile()
    {
        AssetDatabase.Refresh();
        if (EditorApplication.isCompiling)
            yield return new WaitForDomainReload();
        Assert.IsTrue(true);
    }


    [UnityTest]
    public IEnumerator C_EvaluateExperimentGeneration()
    {
        //Debug.Log("Starting GenerateExperimentTest");
        bool success = true;

        // Check #1: Does the new class exist?
        var newExperimentType = ExperimentGenerator.GetSubType(testExperimentName, typeof(Experiment));
        if (newExperimentType == null)
        {
            success = false;
            Debug.LogError("Failure: Dynamic type not found in assembly.");
        }

        // Check #2: Is the appropriate class instance present in the new scene?
        if (GameObject.FindObjectOfType<Experiment>() == null)
        {
            success = false;
            Debug.LogError("Failure: Experiment instance not added to scene.");
        }

        // Check #3: Is the Default UI added to the scene?
        if (GameObject.FindObjectOfType<DefaultUI>() == null)
        {
            success = false;
            Debug.LogError("Failure: Default UI not added to scene.");
        }

        // Check #4: Is the AEPsych Client added to the scene?
        if (GameObject.FindObjectOfType<AEPsychClient>() == null)
        {
            success = false;
            Debug.LogError("Failure: AEPsych Client not added to scene.");
        }
        yield return new WaitForSecondsRealtime(0.5f);

        // Delete the generated script
        string[] targetFolder = { "Assets/Scripts/Experiments"};
        foreach (var asset in AssetDatabase.FindAssets("UnitTestExperiment", targetFolder))
        {
            var path = AssetDatabase.GUIDToAssetPath(asset);
            AssetDatabase.DeleteAsset(path);
        }

        generator = (ExperimentGenerator)EditorWindow.GetWindow(typeof(ExperimentGenerator));
        generator.Close();

        Assert.IsTrue(success);

        yield return null;
    }

}
