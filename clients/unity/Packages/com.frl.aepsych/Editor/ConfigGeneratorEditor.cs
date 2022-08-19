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
using System;

[System.Serializable]
[CustomEditor(typeof(ConfigGenerator)), CanEditMultipleObjects]
public class ConfigGeneratorEditor : Editor
{
    int selected_stimulus;
    int selected_response;
    int selected_method;

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

        selected_stimulus = configGenerator.stimulus_presentation;
        selected_response = configGenerator.response_type;
        selected_method = configGenerator.experiment_method;

        string currConfig = "";
        currConfig += (StimulusType)selected_stimulus + " Stimulus ";
        currConfig += (ResponseType)selected_response + " Input ";
        currConfig += (ExperimentType)selected_method;
        GUILayout.Label(new GUIContent("Current Configuration", "Not all experiment option combinations " +
            "are possible. This Config Generator will only allow you to select combinations for which a " +
            "valid acqisition function & model exist. For more customization, use a manual config file."));
        EditorGUILayout.LabelField(currConfig, new GUIStyle()
        {
            fontSize = 14,
            richText = true,
            wordWrap = true,
            normal = new GUIStyleState() { textColor = Color.yellow }
        });
        EditorGUILayout.Space();

        int temp_select;
        int[] currCombo = new int[] { selected_stimulus, selected_response, selected_method };
        
        GUILayout.Label("Stimulus Type", EditorStyles.boldLabel);
        temp_select = EditorGUILayout.Popup(selected_stimulus, Enum.GetNames(typeof(StimulusType)));
        if (temp_select != selected_stimulus)
        {
            // This value changed this frame
            currCombo = configGenerator.ValidateCombination(temp_select, selected_response, selected_method, 0);
        }
        selected_stimulus = temp_select;
        GUILayout.Space(10f);

        GUILayout.Label("Response Type", EditorStyles.boldLabel);
        temp_select = EditorGUILayout.Popup(selected_response, Enum.GetNames(typeof(ResponseType)));
        if (temp_select != selected_response)
        {
            // This value changed this frame
            currCombo = configGenerator.ValidateCombination(selected_stimulus, temp_select, selected_method, 1);
        }
        selected_response = temp_select;
        GUILayout.Space(10f); //2

        GUILayout.Label("Experiment Type", EditorStyles.boldLabel);
        temp_select = EditorGUILayout.Popup(selected_method, Enum.GetNames(typeof(ExperimentType)));
        if (temp_select != selected_method)
        {
            // This value changed this frame
            currCombo = configGenerator.ValidateCombination(selected_stimulus, selected_response, temp_select, 2);
        }
        selected_method = temp_select;

        configGenerator.stimulus_presentation = currCombo[0];
        configGenerator.response_type = currCombo[1];
        configGenerator.experiment_method = currCombo[2];

        if ((ExperimentType)configGenerator.experiment_method == ExperimentType.Threshold)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Target Threshold:");
            configGenerator.target = EditorGUILayout.FloatField(Mathf.Clamp(configGenerator.target, 0f, 1f));
        }

        configGenerator.GetComponent<Experiment>().stimulusType = (StimulusType) configGenerator.stimulus_presentation;
        configGenerator.GetComponent<Experiment>().responseType = (ResponseType) configGenerator.response_type;

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
