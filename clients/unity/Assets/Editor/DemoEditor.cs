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
       
       // GUILayout.Label("Select Experiment Demo", EditorStyles.boldLabel); //3
        //GUILayout.Space(10f); //2
        GUILayout.Label("Number of parameters", EditorStyles.boldLabel); //3
        string[] options = new string[] { "1", "2", "3" };
        selected_dim = GUILayout.SelectionGrid(selected_dim, options, 1, EditorStyles.radioButton);

        GUILayout.Space(10f); //2
        GUILayout.Label("Method", EditorStyles.boldLabel); //3
        options = new string[] { "Optimization", "Threshold" };
        selected_method = GUILayout.SelectionGrid(selected_method, options, 1, EditorStyles.radioButton);

        GUILayout.Space(20f);
        GUILayout.BeginHorizontal(); //4
        GUILayout.Label("Selected Experiment:", GUILayout.Width(150f)); //5
        selectedScript = GUILayout.TextField(selectedScript); //6
        GUILayout.EndHorizontal(); //7



        if (selected_dim == 0) //1D
        {
            if (selected_method == 0) //optimization
            {
                    selectedScript = "1DSingleOpt";

            }
            else // (selected_method == 1, ie threshold
            {
                selectedScript = "1DSingleDetection";

            }
        }
        else if (selected_dim == 1) //2D
        {
            if (selected_method == 0) //optimization
            {
               
                    selectedScript = "2DSingleOpt";
              
            }
            else // (selected_method == 1, ie threshold
            {
                selectedScript = "2DSingleDetection";

            }
        }
        else if (selected_dim == 2) //3D
        {
            if (selected_method == 0) //optimization
            {
               
                    selectedScript = "3DSingleOpt";
              
            }
            else // (selected_method == 1, ie threshold
            {
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
        serializedObject.FindProperty("m_Method").intValue = selected_method;
        serializedObject.FindProperty("selectedScript").stringValue = selectedScript;

        serializedObject.ApplyModifiedProperties();
    }
}
