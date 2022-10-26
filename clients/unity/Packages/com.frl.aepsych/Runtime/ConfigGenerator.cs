/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using UnityEditor;
using UnityEngine;
using AEPsych;
using System;

namespace AEPsych
{
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
    public enum StimulusType
    {
        Single,
        Pairwise
    }
    public enum ResponseType
    {
        Binary,
        Continuous
    }
    public enum ExperimentType
    {
        Threshold,
        Optimization,
        Exploration
    }
}

[System.Serializable]
#if UNITY_EDITOR
[InitializeOnLoad]
#endif
public class ConfigGenerator: MonoBehaviour
{
    /*
    [SerializeField] public StimulusType stimulus_presentation = StimulusType.Single;
    [SerializeField] public ResponseType response_type = ResponseType.Binary;
    [SerializeField] public ExperimentType experiment_method = ExperimentType.Threshold;
    */
    [SerializeField] public int stimulus_presentation = 0;
    [SerializeField] public int response_type = 0;
    [SerializeField] public int experiment_method = 0;

    [SerializeField] public List<Parameter> experimentParams;

#if UNITY_EDITOR
    [SerializeField] public DefaultAsset manualConfig;
#endif
    [SerializeField] public string relativeFilePath;

    Dictionary<int[], string> valid_combos = new Dictionary<int[], string>
    {
        { new[] { 0, 0, 0 }, "SingleBinaryThreshold" },
        { new[] { 0, 0, 1 }, "SingleBinaryOptimization" },
        { new[] { 0, 1, 0 }, "SingleContinuousThreshold" },
        { new[] { 0, 1, 1 }, "SingleContinuousOptimization" },
        { new[] { 0, 1, 2 }, "SingleContinuousExploration" },
        { new[] { 1, 0, 1 }, "PairwiseBinaryOptimization" },
        { new[] { 1, 0, 2 }, "PairwiseBinaryExploration" }
    };
    public int experiment_idx = 0;
    public float target = 0.5f;
    public int initialization_trials = 10;
    public int optimization_trials = 15;
    public bool isInitialized;
    public bool isAutomatic = true;
    public string fullConfigFile;
    public string experimentName = "";
    public string defaultFilePath = "configs/";
    public bool isEditorVersion = false;
    public bool isValid = true;
    public bool startWithModel;

    private void OnValidate()
    {
        isEditorVersion = true;
        if (experimentParams == null)
        {
            experimentParams = new List<Parameter>();
            experimentParams.Add(new Parameter("Dimension 1"));
        }
#if UNITY_EDITOR
        if (BuildPipeline.isBuildingPlayer)
            return;

        if (manualConfig != null)
        {
            string temp = AssetDatabase.GetAssetPath(manualConfig);
            if (temp.StartsWith(@"Assets/StreamingAssets") || temp.StartsWith(@"Assets\StreamingAssets"))
            {
                // Remove the Assets/StreamingAssets directory from path
                relativeFilePath = temp.Substring(23);
                Debug.Log("manual config: " + relativeFilePath);
            }
            else
            {
                Debug.LogError("Manual Config must be moved under Assets/StreamingAssets. Current location: " + temp);
            }
        }
#endif
    }

    string FindNameFromIndicies(int[] combo)
    {
        foreach (int[] c in valid_combos.Keys.ToArray())
        {
            if (c.SequenceEqual(combo))
            {
                return valid_combos[c];
            }
        }
        return null;
    }

    // Try to fetch a valid combination that retains the
    // value of the last-changed field
    public int[] ValidateCombination(int s, int r, int m, int lastChanged)
    {
        int stimulus = s;
        int response = r;
        int method = m;

        int s_Count = Enum.GetNames(typeof(StimulusType)).Length;
        int r_Count = Enum.GetNames(typeof(ResponseType)).Length;
        int m_Count = Enum.GetNames(typeof(ExperimentType)).Length;

        // output array
        int[] combo = new[] { (stimulus), (response), (method) };

        // If already valid, do nothing
        if (FindNameFromIndicies(combo) != null)
            return combo;

        // value of last changed field
        int lastChangedVal = combo[lastChanged];

        // Depth First Search for different values until valid
        // Starting search at curr values, not 0, to ensure minimum field modification
        for (int i = 0; i < s_Count; i++)
        {
            combo[0] = stimulus;
            for(int j = 0; j < r_Count; j++)
            {
                combo[1] = response;
                for (int k = 0; k < m_Count; k++)
                {
                    combo[2] = method;

                    // ensure last modified value remains intact 
                    combo[lastChanged] = lastChangedVal;

                    // Check combo validity
                    if (FindNameFromIndicies(combo) != null)
                    {
                        return combo;
                    }

                    // Increment & wrap-around
                    method = (method + 1) % m_Count;
                }
                // Increment & wrap-around
                response = (response + 1) % r_Count;
            }
            // Increment & wrap-around
            stimulus = (stimulus + 1) % s_Count;
        }

        Debug.LogError("No valid combos found.");
        return new int[] { s, r, m };
    }

    public void SetName(string newName)
    {
        experimentName = newName;
    }

    public string GetManualConfigPath()
    {
        return relativeFilePath;
    }

    public virtual string[] GetExperimentTypes()
    {
        return valid_combos.Values.ToArray();
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
        int num_stimuli = 1;
        int inducing_size;
        int restarts;
        int samps;
        float beta = 0f;
        bool specifyAcqf = false;

        switch (FindNameFromIndicies(new[] { stimulus_presentation, response_type, experiment_method }))
        {
            case "SingleBinaryThreshold":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "binary";
                acqf = "MCLevelSetEstimation";
                model = "GPClassificationModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                beta = 3.98f;
                specifyAcqf = true;
                break;
            case "SingleBinaryOptimization":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "binary";
                acqf = "qNoisyExpectedImprovement";
                model = "GPClassificationModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                target = -1f;
                specifyAcqf = true;
                break;
            case "SingleContinuousThreshold":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "continuous";
                acqf = "MCLevelSetEstimation";
                model = "GPRegressionModel";
                specifyAcqf = true;
                objective = "IdentityMCObjective";
                break;
            case "SingleContinuousOptimization":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "continuous";
                acqf = "qNoisyExpectedImprovement";
                model = "GPRegressionModel";
                target = -1f;
                specifyAcqf = false;
                break;
            case "SingleContinuousExploration":
                init_generator = "SobolGenerator";
                opt_generator = "OptimizeAcqfGenerator";
                outcome_type = "continuous";
                acqf = "MCPosteriorVariance";
                model = "GPRegressionModel";
                target = -1f;
                specifyAcqf = false;
                break;
            case "PairwiseBinaryOptimization":
                init_generator = "PairwiseSobolGenerator";
                opt_generator = "PairwiseOptimizeAcqfGenerator";
                outcome_type = "binary";
                acqf = "qNoisyExpectedImprovement";
                model = "PairwiseProbitModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                target = -1f;
                num_stimuli = 2;
                specifyAcqf = true;
                break;
            case "PairwiseBinaryExploration":
                init_generator = "PairwiseSobolGenerator";
                opt_generator = "PairwiseOptimizeAcqfGenerator";
                outcome_type = "binary";
                acqf = "PairwiseMCPosteriorVariance";
                model = "PairwiseProbitModel";
                mean_covar_factory = "default_mean_covar_factory";
                objective = "ProbitObjective";
                target = -1f;
                num_stimuli = 2;
                specifyAcqf = true;
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
        sb.Append("stimuli_per_trial = " + num_stimuli + "\n");
        sb.Append("outcome_types = [" + outcome_type + "]\n");
        if (target >= 0)
            sb.Append("target = " + target + "\n");
        sb.Append("\n");
        sb.Append("strategy_names = [init_strat, opt_strat]\n");
        sb.Append("\n");

        // Initial Strategy
        sb.Append("[init_strat]\n");
        sb.Append("min_asks = " + initialization_trials + "\n");
        sb.Append("generator = " + init_generator + "\n");
        if (startWithModel)
        {
            sb.Append("model = " + model + "\n");
        }
        sb.Append("\n");

        // Optimized Strategy
        sb.Append("[opt_strat]\n");
        sb.Append("min_asks = " + optimization_trials + "\n");
        sb.Append("refit_every = 5\n");
        sb.Append("generator = " + opt_generator + "\n");
        sb.Append("acqf = " + acqf + "\n");
        sb.Append("model = " + model + "\n");
        sb.Append("\n");

        // Model Parameters
        sb.Append("[" + model + "]\n");
        sb.Append("inducing_size = " + inducing_size + "\n");
        if (mean_covar_factory != "")
            sb.Append("mean_covar_factory = " + mean_covar_factory + "\n");
        sb.Append("\n");

        // Optimized Generator Parameters
        sb.Append("[" + opt_generator + "]\n");
        sb.Append("restarts = " + restarts + "\n");
        sb.Append("samps = " + samps + "\n");
        sb.Append("\n");

        if (specifyAcqf)
        {
            // Acquisition Function Parameters
            sb.Append("[" + acqf + "]\n");
            if (beta != 0f)
                sb.Append("beta = " + beta + "\n");
            sb.Append("objective = " + objective + "\n");
        }
        
        return sb.ToString();
    }

#if UNITY_EDITOR
    public void WriteToFile()
    {
        fullConfigFile = GetConfigText();
        string finalPath;
        string relativePath = "";
        if (manualConfig == null)
        {
            finalPath = Path.Combine(Application.streamingAssetsPath, "configs/" + experimentName + ".ini");
            relativePath = Path.Combine("Assets/StreamingAssets/configs/" + experimentName + ".ini");
        }
            
        else
            finalPath = Path.Combine(Application.streamingAssetsPath, GetManualConfigPath());

        if (!Directory.Exists(Path.GetDirectoryName(finalPath)))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(finalPath));
        }
        Debug.Log("final path: " + finalPath);
        StreamWriter writer = new StreamWriter(finalPath, false);
        writer.WriteLine(fullConfigFile);
        writer.Close();
        AssetDatabase.Refresh();
        if (manualConfig == null)
            manualConfig = AssetDatabase.LoadAssetAtPath<DefaultAsset>(relativePath);
        Debug.Log("Wrote config to " + finalPath);
    }
#endif
}
