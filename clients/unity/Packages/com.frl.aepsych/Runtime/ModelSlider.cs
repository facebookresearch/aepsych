using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AEPsych;
using TMPro;
using UnityEngine.UI;

public class ModelSlider : MonoBehaviour
{
    public TextMeshProUGUI minLabel;
    public TextMeshProUGUI maxLabel;
    public TextMeshProUGUI nameLabel;
    public TextMeshProUGUI valueLabel;

    QueryModel queryModel;
    Slider slider;
    public bool isLocked = false;
    string sliderName;
    float value;
    public bool isY = false;

    public void ToggleLocked()
    {
        isLocked = !isLocked;
    }
    public void InitSlider(float min, float max, string name, bool y)
    {
        slider = GetComponentInChildren<Slider>(true);
        this.gameObject.name = name + "Slider";
        minLabel.text = min.ToString("F3");
        maxLabel.text = max.ToString("F3");
        nameLabel.text = name;
        slider.minValue = min;
        slider.maxValue = max;
        SetValue(min);
        this.sliderName = name;
        this.isY = y;

    }

    public void SetQueryModel(QueryModel q)
    {
        queryModel = q;
    }

    public void SetValueFromHandle()
    {
        float v = slider.value;
        this.value = v;
        valueLabel.text = v.ToString("F3");

        // Prevent duplicate requests by checking if client is busy
        if (queryModel.client.IsBusy())
        {
            return;
        }

        if (!isY)
        {
            StartCoroutine(queryModel.ComputeYFromModel());
        }
        else
        {
            StartCoroutine(queryModel.ComputeInverseFromModel());
        }
    }

    public void DisableLockToggle()
    {
        GetComponentInChildren<Toggle>().gameObject.SetActive(false);
    }

    public void SetValue(float v)
    {
        slider.value = v;
        this.value = v;
        valueLabel.text = v.ToString("F3");
    }
    public float GetValue()
    {
        return value;
    }

    public string GetName()
    {
        return sliderName;
    }
}
