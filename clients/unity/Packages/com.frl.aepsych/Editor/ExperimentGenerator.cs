/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;
using System;
using UnityEngine;
using System.Reflection;
using AEPsych;
using TMPro;
using UnityEngine.EventSystems;

public class ExperimentGenerator : EditorWindow
{
    #region Experiment Fields

    public string experimentName = "Default";

    #endregion

    public bool needToAttach = false;
    public float waitForCompile = 1;
    public string fileDestination = "/Scripts/Experiments";
    public GameObject tempExperiment;
    public GameObject ExperimentUIPrefab;
    public string prefabPath = "Packages/com.frl.aepsych/Runtime/Prefabs/ExperimentUI.prefab";

    [MenuItem("AEPsych/Create New Experiment")]
    public static void CreateNew()
    {
        EditorWindow.GetWindow(typeof(ExperimentGenerator));
    }

    void OnGUI()
    {
        GUILayout.Label("Name your Experiment Script. \n\nName must be unique - a new class with this name \nwill be generated.");
        GUILayout.Space(10);

        GUILayout.BeginHorizontal();
        GUILayout.Label("Experiment Name", new GUILayoutOption[0]);
        experimentName = EditorGUILayout.TextField(experimentName, new GUILayoutOption[0]);
        GUILayout.EndHorizontal();
        GUILayout.Space(10);
        GUI.color = Color.green;

        if (GUILayout.Button("Create", new GUILayoutOption[0]))
            CreateNewExperiment();

    }

    void Update()
    {
        if (needToAttach)
        {
            waitForCompile -= 0.01f;
            EditorUtility.DisplayProgressBar("AEPsych Client", "Generating new experiment script...", 1 - waitForCompile);
            if (waitForCompile <= 0)
            {
                if (!EditorApplication.isCompiling)
                {
                    tempExperiment = new GameObject();
                    tempExperiment.name = experimentName;
                    var experimentType = GetSubType(experimentName.Replace(" ", ""), typeof(Experiment));
                    if (experimentType != null)
                    {
                        tempExperiment.AddComponent(experimentType);
                    }
                    else
                    {
                        Debug.LogError(string.Format("Unable to find dynamically generated Type: {0}. Please manually assign the {0}.cs script onto the {0} gameobject in the scene.", experimentName));
                    }
                    needToAttach = false;
                    waitForCompile = 1;
                    tempExperiment.AddComponent<ConfigGenerator>().SetName(experimentName);
                    EditorUtility.ClearProgressBar();
                }
            }
        }
    }

    private void CreateNewExperiment()
    {
        TextAsset template = AssetDatabase.LoadAssetAtPath("Packages/com.frl.aepsych/Editor/Templates/ExperimentTemplate.txt", typeof(TextAsset)) as TextAsset;

        string contents = "";
        if (template != null)
        {
            contents = template.text;
            contents = contents.Replace("EXPERIMENTCLASS_NAME", experimentName.Replace(" ", ""));
            contents = contents.Replace("EXPERIMENT_NAME", experimentName);
        }
        else
        {
            Debug.LogError("Packages/com.frl.aepsych/Editor/Templates/ExperimentTemplate.txt is Missing!");
        }

        if (!Directory.Exists(Application.dataPath + fileDestination))
        {
            Directory.CreateDirectory(Application.dataPath + fileDestination);
        }

        using (StreamWriter sw = new StreamWriter(string.Format(Application.dataPath + fileDestination + "/{0}.cs",
                                                           new object[] { experimentName.Replace(" ", "") })))
        {
            sw.Write(contents);
        }
        //Refresh the Asset Database
        AssetDatabase.Refresh();

        needToAttach = true;

        // Add an AEPsychClient object if there is not one already
        if (FindObjectOfType<AEPsychClient>() == null)
        {
            AEPsychClient clientObj = new GameObject("AEPsychClient").AddComponent<AEPsychClient>();

            // Try to attach auto start server component, if it exists. (FB internal access only)
            var serverAutoStart = GetSubType("ServerAutoStart", typeof(MonoBehaviour));
            if (serverAutoStart != null)
            {
                clientObj.gameObject.AddComponent(serverAutoStart);
            }
        }

        // Add a DefaultUI object if there is not one already
        if (FindObjectOfType<DefaultUI>(true) == null)
        {
            DefaultUI UIObj;
            ExperimentUIPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (ExperimentUIPrefab != null)
            {
                UIObj = Instantiate(ExperimentUIPrefab).GetComponent<DefaultUI>();
                UIObj.gameObject.name = "Experiment UI";
                if (FindObjectOfType<EventSystem>() == null)
                {
                    GameObject eventSystem = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
                }
            }
            else
                Debug.LogError(string.Format("Prefab not found at: {0}. Please Manually add the ExperimentUI prefab to your scene.", prefabPath));

            //DefaultUI UIObj = new GameObject("Default UI").AddComponent<DefaultUI>();

        }
    }

    public static Type GetSubType(string typeName, Type baseClass)
    {
        Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
        foreach (var asm in assemblies)
        {
            Type[] types = asm.GetTypes();
            foreach (var T in types)
            {
                if (T.IsSubclassOf(baseClass))
                {
                    if (T.FullName == typeName)
                    {
                        return T;
                    }
                }
            }
        }
        return null;
    }

    public static GameObject LoadPrefab(string name)
    {
        return AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(AssetDatabase.FindAssets(name)[0]));
    }
}
