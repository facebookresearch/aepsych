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
using UnityEditor;

[System.Serializable]
// Declare type of Custom Editor
[CustomEditor(typeof(DemoSelector)), CanEditMultipleObjects] //1
public class DemoEditor : Editor
{
    int selected_dim = 0;
    int selected_response = 0;
    int selected_method = 0;
    string selectedScript = "None";
    // OnInspector GUI
    public override void OnInspectorGUI() //2
    {
        if (Application.isPlaying)
        {
            return;
        }
        selected_dim = serializedObject.FindProperty("m_Dim").intValue;
        selected_response = serializedObject.FindProperty("m_Response").intValue;
        selected_method = serializedObject.FindProperty("m_Method").intValue;
        selectedScript = serializedObject.FindProperty("selectedScript").stringValue;
        base.OnInspectorGUI();
        GUILayout.Label("Number of parameters", EditorStyles.boldLabel); //3
        string[] options = new string[] { "1", "2", "3" };
        selected_dim = GUILayout.SelectionGrid(selected_dim, options, 1, EditorStyles.radioButton);
        GUILayout.Space(10f); //2
        GUILayout.Label("Method", EditorStyles.boldLabel); //3
        options = new string[] { "Optimization", "Threshold", "Exploration" };
        selected_method = GUILayout.SelectionGrid(selected_method, options, 1, EditorStyles.radioButton);
        //threshold is single only
        if (selected_method == 1)
        {
            selected_response = 0;
        }
        //exploration is pairwise only
        if (selected_method == 2)
        {
            selected_response = 1;
        }
        GUILayout.Space(10f); //2
        GUILayout.Label("Stimulus presentation", EditorStyles.boldLabel); //3
        options = new string[] { "Single", "Pairwise" };
        selected_response = GUILayout.SelectionGrid(selected_response, options, 1, EditorStyles.radioButton);
        GUILayout.Space(20f);
        GUILayout.BeginHorizontal(); //4
        GUILayout.Label("Selected Experiment:", GUILayout.Width(150f)); //5
        selectedScript = GUILayout.TextField(selectedScript); //6
        GUILayout.EndHorizontal(); //7
        if (selected_dim == 0) //1D
        {
            if (selected_method == 0) //optimization
            {
                if (selected_response == 0) //single
                {
                    selectedScript = "1DSingleOpt";
                }
                else //pairwise
                {
                    selectedScript = "1DPairwiseOpt";
                }
            }
            else if (selected_method == 1) // threshold
            {
                //selected_response=0, single only
                selectedScript = "1DSingleDetection";
            }
            else if (selected_method == 2) //exploration
            {
                selectedScript = "1DPairwiseExploration";
            }
        }
        else if (selected_dim == 1) //2D
        {
            if (selected_method == 0) //optimization
            {
                if (selected_response == 0) //single
                {
                    selectedScript = "2DSingleOpt";
                }
                else //pairwise
                {
                    selectedScript = "2DPairwiseOpt";
                }
            }
            else if (selected_method == 2) //exploration
            {
                selectedScript = "2DPairwiseExploration";
            }
            else // (selected_method == 1, ie threshold
            {
                //selected_response=0, single only
                selectedScript = "2DSingleDetection";
            }
        }
        else if (selected_dim == 2) //3D
        {
            if (selected_method == 0) //optimization
            {
                if (selected_response == 0) //single
                {
                    selectedScript = "3DSingleOpt";
                }
                else //pairwise
                {
                    selectedScript = "3DPairwiseOpt";
                }
            }
            else if (selected_method == 2) //exploration
            {
                selectedScript = "3DPairwiseExploration";
            }
            else // (selected_method == 1, ie threshold
            {
                //selected_response=0, single only
                selectedScript = "3DSingleDetection";
            }
        }
        Transform parentTransform = Selection.activeGameObject.transform;
        for (int j = 0; j < parentTransform.childCount; j++)
        {
            parentTransform.GetChild(j).gameObject.SetActive(false);
        }
        parentTransform.Find(selectedScript).gameObject.SetActive(true);

        serializedObject.FindProperty("m_Dim").intValue = selected_dim;
        serializedObject.FindProperty("m_Response").intValue = selected_response;
        serializedObject.FindProperty("m_Method").intValue = selected_method;
        serializedObject.FindProperty("selectedScript").stringValue = selectedScript;
        serializedObject.ApplyModifiedProperties();
    }
}
