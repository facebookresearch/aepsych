/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using AEPsych;

[System.Serializable]
[CustomEditor(typeof(ConfigGenerator)), CanEditMultipleObjects]
public class ConfigGeneratorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        ConfigGenerator configGenerator = (ConfigGenerator)serializedObject.targetObject;

        if (configGenerator.experimentParams.Count == 0)
        {
            configGenerator.experimentParams.Add(new Parameter("Dimension 1"));
        }
        configGenerator.isAutomatic = EditorGUILayout.ToggleLeft("Use Automatic Config Generation", configGenerator.isAutomatic);
        EditorGUILayout.Space();
        if (!configGenerator.isAutomatic)
        {
            EditorGUILayout.LabelField("Manual config reference (should be within the StreamingAssets folder)", new GUIStyle()
            {
                fontSize = 18,
                richText = true,
                wordWrap = true,
                normal = new GUIStyleState() { textColor = Color.white }
            }); ;
            var experimentConfig = serializedObject.FindProperty("manualConfig");
            EditorGUILayout.PropertyField(experimentConfig, new GUIContent("Manual Config File"), true);

            EditorGUILayout.Space();

            EditorGUILayout.HelpBox("Configure the following properties, then Click \"Write Config to File\" to overwrite the config file specified above. If null, the new config will default to " + configGenerator.defaultFilePath, MessageType.Info);
            if (GUILayout.Button("Write Config to File"))
            {
                if (configGenerator.experimentParams != null && configGenerator.experimentParams.Count != 0)
                {
                    configGenerator.WriteToFile();
                }
                else
                {
                    Debug.LogError("Cannot write a config file with zero dimensions!");
                }
            }
            EditorGUILayout.Space();
        }
        EditorGUILayout.Space();
        configGenerator.initialization_trials = EditorGUILayout.IntField("Number of initialization trials", configGenerator.initialization_trials);
        configGenerator.optimization_trials = EditorGUILayout.IntField("Number of optimization trials", configGenerator.optimization_trials);
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Experiment Type:");
        configGenerator.experiment_idx = EditorGUILayout.Popup(configGenerator.experiment_idx, configGenerator.GetExperimentTypes());
        if (configGenerator.GetExperimentTypes()[configGenerator.experiment_idx].Contains("Threshold"))
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Target Threshold:");
            configGenerator.target = EditorGUILayout.FloatField(Mathf.Clamp(configGenerator.target, 0f, 1f));
        }
        EditorGUILayout.Space();
        configGenerator.startWithModel = EditorGUILayout.ToggleLeft("Initialize Model on Start", configGenerator.startWithModel);
        EditorGUILayout.Space();
        var experimentParams = serializedObject.FindProperty("experimentParams");
        EditorGUILayout.PropertyField(experimentParams, new GUIContent("Experiment Dimensions"), true);
        serializedObject.ApplyModifiedProperties();
        if (GUI.changed) {
            EditorUtility.SetDirty(configGenerator);
            // Ensure experiment name is transfered in the case of manual experiment creation
            if (configGenerator.experimentName == "")
            {
                configGenerator.SetName(configGenerator.GetComponent<Experiment>().GetName());
            }
        }
    }
}
