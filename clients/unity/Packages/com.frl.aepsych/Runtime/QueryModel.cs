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
using AEPsych;
using UnityEngine.UI;

public class QueryModel : MonoBehaviour
{
    [SerializeField] public string yAxisName = "y-axis";
    public AEPsychClient client;
    public GameObject sliderPrefab;
    public GameObject xslidergroup;
    public CanvasGroup canvasGroup;
    public Experiment experiment;
    public Button toggleButton;

    List<ModelSlider> xSliders = new List<ModelSlider>();
    Canvas[] otherCanvases = { };
    ModelSlider ySlider;
    DefaultUI defaultUI;
    bool initialized = false;
    bool probSpace = false;
    TrialConfig initParams;
    TrialConfig currConfig;
    float maxY;
    float minY;

    // Start is called before the first frame update
    void Start()
    {
        if (client == null)
        {
            client = FindObjectOfType<AEPsychClient>();
        }
        if (experiment == null)
        {
            experiment = FindObjectOfType<Experiment>();
        }
        defaultUI = GetComponent<DefaultUI>();

        if (!experiment.UsesModelExplorer())
            toggleButton.gameObject.SetActive(false);

        if (GetComponent<Canvas>().worldCamera == null)
        {
            GetComponent<Canvas>().worldCamera = FindObjectOfType<Camera>();
        }

        if (experiment.autoDisableOtherCanvases)
        {
            otherCanvases = FindObjectsOfType<Canvas>();
        }
    }

    public List<float> GetSliderValues()
    {
        List<float> values = new List<float>();
        foreach (ModelSlider param in xSliders)
        {
            values.Add(param.GetValue());
        }
        return values;
    }

    public IEnumerator SpawnSliders()
    {
        QueryEnabled(false);
        //Get the current config with n parameters
        yield return StartCoroutine(client.Params());
        //initParams = new TrialConfig();
        //initParams.Add("Dimension 1", new List<float>() { 0.0f });
        initParams = client.GetConfig(version: "0.0");
        if (!initialized)
        {
            foreach (KeyValuePair<string, List<float>> entry in initParams)
            {
                string name = entry.Key;
                float min = entry.Value[0];
                float max = entry.Value[1];
                GameObject sliderObj = Instantiate(sliderPrefab, canvasGroup.transform.position, canvasGroup.transform.rotation, canvasGroup.transform);
                sliderObj.transform.parent = xslidergroup.transform;
                ModelSlider slider = sliderObj.GetComponent<ModelSlider>();
                slider.InitSlider(min, max, name, false);
                slider.SetQueryModel(this);
                xSliders.Add(slider);
            }
        }
        //Spawn final slider and query twice for its limits

        yield return StartCoroutine(client.Query(QueryType.max));
        QueryMessage m = client.GetQueryResponse();
        maxY = m.y;
        yield return StartCoroutine(client.Query(QueryType.min));
        m = client.GetQueryResponse();
        minY = m.y;

        string nameY = yAxisName;

        //Spawn + initialize slider, if necessary.
        GameObject ySliderObj;
        if (!initialized)
        {
            ySliderObj = Instantiate(sliderPrefab, canvasGroup.transform.position, canvasGroup.transform.rotation, canvasGroup.transform);
            RectTransform rt = ySliderObj.GetComponent<RectTransform>();
            rt.localScale *= 1.25f;
            rt.anchorMax = new Vector2(.9f, 0.55f);
            rt.anchorMin = new Vector2(.9f, 0.55f);
            rt.pivot = new Vector2(1f, 0.5f);
            ySlider = ySliderObj.GetComponent<ModelSlider>();
            ySlider.SetQueryModel(this);
            ySlider.DisableLockToggle();
        }

        //ySlider.InitSlider(0, 3.4567f, nameY, true);
        ySlider.InitSlider(minY, maxY, nameY, true);
        StartCoroutine(ComputeYFromModel());
        initialized = true;
    }

    public void ToggleProbabilitySpace()
    {
        probSpace = !probSpace;
        if (probSpace)
        {
            ySlider.InitSlider(0, 1, yAxisName, true);
        }
        else
        {
            ySlider.InitSlider(minY, maxY, yAxisName, true);
        }
        SetY();
    }

    public bool IsInitialized()
    {
        return initialized;
    }

    public void SetY()
    {
        StartCoroutine(ComputeYFromModel());
    }

    public IEnumerator ComputeYFromModel()
    {
        QueryEnabled(false);
        TrialConfig x = new TrialConfig();
        foreach (ModelSlider slider in xSliders)
        {
            x.Add(slider.GetName(), new List<float>() { slider.GetValue() });
        }
        yield return StartCoroutine(client.Query(QueryType.prediction, x: x, probability_space: probSpace));
        QueryMessage m = client.GetQueryResponse();
        ySlider.SetValue(m.y);
        QueryEnabled(true);
    }

    public IEnumerator ComputeInverseFromModel()
    {
        QueryEnabled(false);
        float y = ySlider.GetValue();
        TrialConfig constraints = new TrialConfig();
        foreach (ModelSlider slider in xSliders)
        {
            if (slider.isLocked)
            {
                constraints.Add(slider.GetName(), new List<float>() { slider.GetValue() });
            }
        }
        yield return StartCoroutine(client.Query(QueryType.inverse, constraints: constraints, y: y, probability_space: probSpace));
        QueryMessage m = client.GetQueryResponse();
        foreach (ModelSlider slider in xSliders)
        {
            slider.SetValue(m.x[slider.GetName()][0]);
        }
        ySlider.SetValue(m.y);
        QueryEnabled(true);
    }

    public void ShowQueryResults()
    {
        QueryEnabled(false);
        TrialConfig queryConfig = new TrialConfig();
        List<float> values = GetSliderValues();
        int idx = 0;
        foreach (KeyValuePair<string, List<float>> entry in initParams)
        {
            queryConfig[entry.Key] = new List<float>();
            // Add value to 1st and 2nd index of List, in the case that the ShowStimuli
            // method of the experiment reads two values for each dimension
            queryConfig[entry.Key].Add(values[idx]);
            queryConfig[entry.Key].Add(values[idx]);
            idx++;
        }

        HideSliders();
        experiment.ShowStimuli(queryConfig);
    }

    public void HideSliders()
    {
        canvasGroup.blocksRaycasts = false;
        canvasGroup.interactable = false;
        canvasGroup.alpha = 0;
        if (defaultUI != null)
            defaultUI.SetTextActive(true);
    }

    public void ShowSliders()
    {
        canvasGroup.blocksRaycasts = true;
        canvasGroup.interactable = true;
        canvasGroup.alpha = 1;
        if (defaultUI != null)
            defaultUI.SetTextActive(false);
    }

    public void ToggleSliders()
    {
        bool isQueryMode = canvasGroup.blocksRaycasts;
        // Enable/Disable other canvases to prevent raycast blocking
        foreach (Canvas c in otherCanvases)
        {
            if (c.GetComponentInChildren<QueryModel>() == null)
                c.gameObject.SetActive(isQueryMode);
        }

        if (isQueryMode)
        {
            experiment.StopExploration();
            HideSliders();
        }
        else
        {
            experiment.StartExploration();
            StartCoroutine(SpawnSliders());
            ShowSliders();
        }
    }

    public void QueryEnabled(bool val)
    {
        toggleButton.interactable = val;
        canvasGroup.interactable = val;
    }

}
