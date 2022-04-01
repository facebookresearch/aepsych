using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using AEPsych;
using UnityEditor.SceneManagement;

[System.Serializable]
public class Parameter
{
    public float lowerBound;
    public float upperBound;
    public string name;

    public Parameter(string _name)
    {
        name = _name;
        lowerBound = 0;
        upperBound = 1;
    }
}

[System.Serializable]
[InitializeOnLoad]
public class ConfigGenerator: MonoBehaviour
{

    [SerializeField] public DefaultAsset manualConfigFile;

    [SerializeField] public List<Parameter> experimentParams;
    public string filePath = "configs";
    public string fileName = "default_config.ini";
    string[] experiment_types = { "Threshold", "Optimization"};
    public int experiment_idx = 0;
    public float target = 0.5f;
    public int initialization_trials = 10;
    public int optimization_trials = 15;
    public bool isInitialized;
    public bool isAutomatic = true;
    public string fullConfigFile;
    public string experimentName = "";
    public bool isEditorVersion = false;
    public bool isValid = true;

    public ConfigGenerator()
    {
#if UNITY_EDITOR
        EditorApplication.playModeStateChanged += ModeStateChanged;
#endif
    }

    public void SetName(string newName)
    {
        experimentName = newName;
    }

    public virtual string[] GetExperimentTypes()
    {
        return experiment_types;
    }

    public void ModeStateChanged(PlayModeStateChange state)
    {
        // This callback is triggered mutliple times, so it's necessary to ignore
        // duplicate calls from default-instance versions of this script
        if (state == PlayModeStateChange.ExitingEditMode)
        {
            if (!isEditorVersion || !this)
                return;

            GenerateConfig();
        }
    }

    public void GenerateConfig()
    {
        if (!isAutomatic)
        {
            PlayerPrefs.SetString(experimentName + "config", "Manual");
            return;
        }

        // Try to cast to Prerelease config

        if (experimentParams != null && experimentParams.Count != 0)
        {
            PlayerPrefs.SetString(experimentName + "config", GetConfigText());
            fullConfigFile = GetConfigText();
        }
        else
        {
            PlayerPrefs.SetString(experimentName + "config", "Empty");
        }
    }

    private void OnValidate()
    {
        isEditorVersion = true;
        if (experimentParams == null)
        {
            experimentParams = new List<Parameter>();
            experimentParams.Add(new Parameter("Dimension 1"));
        }
        
    }

    public bool CheckParamValidity()
    {
        List<string> keys = new List<string>();
        foreach (Parameter p in experimentParams)
        {
            if (keys.Contains(p.name))
            {
                Debug.LogError(string.Format("Found duplicate Dimension name: {0}. Dimension names must all be unique.", p.name));
                return false;
            }
            keys.Add(p.name);
            if (p.upperBound < p.lowerBound)
            {
                Debug.LogError(string.Format("Upper bound must be greater than lower bound in dimension: {0}.", p.name));
                return false;
            }
        }
        return true;
    }

    public virtual string GetConfigText()
    {
        string names = "";
        string lbs = "";
        string ubs = "";
        string outcome_type = "";
        string acqf = "";
        string objective = "";
        string init_generator = "";
        string opt_generator = "";
        string model = "";
        string mean_covar_factory = "";
        int inducing_size;
        int restarts;
        int samps;
        float beta = 0f;

        switch (GetExperimentTypes()[experiment_idx])
        {
            case "Threshold":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "single_probit";
                acqf = "MCLevelSetEstimation";
                model = "GPClassificationModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                beta = 3.98f;
                break;
            case "Optimization":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "single_probit";
                acqf = "qNoisyExpectedImprovement";
                model = "GPClassificationModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                target = -1f;
                break;
            default:
                break;
        }

        foreach (Parameter param in experimentParams)
        {
            names += param.name + ", ";
            lbs += param.lowerBound + ", ";
            ubs += param.upperBound + ", ";
        }

        // Trim the trailing comma
        names = names.Remove(names.Length - 2);
        lbs = lbs.Remove(lbs.Length - 2);
        ubs = ubs.Remove(ubs.Length - 2);
        

        // Scale some parameters based on number of dimensions
        samps = 500 * (experimentParams.Count + 1);
        inducing_size = 50 * experimentParams.Count;
        restarts = 10;

        StringBuilder sb = new StringBuilder();

        // Common Attributes
        sb.Append("[common]\n");
        sb.Append("parnames = [" + names + "]\n");
        sb.Append("lb = [" + lbs + "]\n");
        sb.Append("ub = [" + ubs + "]\n");
        sb.Append("outcome_type = " + outcome_type + "\n");
        if (target >= 0)
            sb.Append("target = " + target + "\n");
        sb.Append("\n");
        sb.Append("strategy_names = [init_strat, opt_strat]\n");
        sb.Append("\n");

        // Initial Strategy
        sb.Append("[init_strat]\n");
        sb.Append("n_trials = " + initialization_trials + "\n");
        sb.Append("generator = " + init_generator + "\n");
        sb.Append("\n");

        // Optimized Strategy
        sb.Append("[opt_strat]\n");
        sb.Append("n_trials = " + optimization_trials + "\n");
        sb.Append("refit_every = 5\n");
        sb.Append("generator = " + opt_generator + "\n");
        sb.Append("acqf = " + acqf + "\n");
        sb.Append("model = " + model + "\n");
        sb.Append("\n");

        // Model Parameters
        sb.Append("[" + model + "]\n");
        sb.Append("inducing_size = " + inducing_size + "\n");
        sb.Append("mean_covar_factory = " + mean_covar_factory + "\n");
        sb.Append("\n");

        // Optimized Generator Parameters
        sb.Append("[" + opt_generator + "]\n");
        sb.Append("restarts = " + restarts + "\n");
        sb.Append("samps = " + samps + "\n");
        sb.Append("\n");

        // Acquisition Function Parameters
        sb.Append("[" + acqf + "]\n");
        if (beta != 0f)
            sb.Append("beta = " + beta + "\n");
        sb.Append("objective = " + objective + "\n");

        return sb.ToString();
    }

    public void WriteToFile()
    {
        fullConfigFile = GetConfigText();
        string finalPath = Path.Combine(Application.streamingAssetsPath, "..", filePath);
        if (!Directory.Exists(finalPath))
        {
            Directory.CreateDirectory(finalPath);
        }
        finalPath = Path.Combine(finalPath, fileName);
        StreamWriter writer = new StreamWriter(finalPath, false);
        writer.WriteLine(fullConfigFile);
        writer.Close();
        AssetDatabase.Refresh();
        Debug.Log("Wrote config to " + finalPath);
    }
}
