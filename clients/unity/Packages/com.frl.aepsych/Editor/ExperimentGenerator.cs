using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;
using System;
using UnityEngine;
using System.Reflection;
using AEPsych;
using TMPro;

// From StackOverflow: https://stackoverflow.com/questions/27802185/unity-4-6-editor-create-a-script-with-predefined-data
public class ExperimentGenerator : EditorWindow
{
    #region Experiment Fields

    public string experimentName = "Default";

    #endregion
    
    private bool needToAttach = false;
    private float waitForCompile = 1;
    private string fileDestination = "/Scripts/Experiments";
    GameObject tempExperiment;

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
        }

        // Add a DefaultUI object if there is not one already
        if (FindObjectOfType<DefaultUI>() == null)
        {
            DefaultUI UIObj = new GameObject("Default UI").AddComponent<DefaultUI>();
            UIObj.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;

            GameObject textObj = new GameObject("Trial Text");
            textObj.transform.SetParent(UIObj.transform);

            TextMeshProUGUI experimentText = textObj.AddComponent<TextMeshProUGUI>();
            experimentText.alignment = TextAlignmentOptions.Center;
            experimentText.outlineColor = Color.black;
            experimentText.outlineWidth = 0.2f;

            RectTransform rect = textObj.GetComponent<RectTransform>();
            rect.anchoredPosition = Vector2.zero;
            rect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 500f);
            rect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, 200f);
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
}
