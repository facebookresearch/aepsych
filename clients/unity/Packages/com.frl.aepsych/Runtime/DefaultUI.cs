/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using AEPsych;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(RectTransform), typeof(Canvas))]
public class DefaultUI : MonoBehaviour
{
    public bool enableTrialReplay = true;
    public bool enableExperimentRestart = true;
    public TextMeshProUGUI experimentText;
    public GameObject foldoutMenu;
    public GameObject foldoutMenuIcon;
    public GameObject responseSlider;
    Experiment experiment;
    Button[] menuButtons;

    private void Start()
    {
        menuButtons = foldoutMenu.GetComponentsInChildren<Button>();
        foreach (Button b in menuButtons)
        {
            if (b.gameObject.name == "ReplayButton")
            {
                if (!enableTrialReplay)
                    b.gameObject.SetActive(false);
            }
            if (b.gameObject.name == "RestartButton")
            {
                if (!enableExperimentRestart)
                    b.gameObject.SetActive(false);
            }
        }
        
    }

    public void SetText(string msg)
    {
        if (experimentText != null)
            experimentText.text = msg;
    }

    public void SetTextActive(bool active)
    {
        experimentText.enabled = active;
    }

    public void ShowResponseSlider()
    {
        responseSlider.SetActive(true);
        responseSlider.GetComponent<Slider>().value = 0.5f;
    }

    public void HideResponseSlider()
    {
        responseSlider.SetActive(false);
    }

    public void SumbitSliderResponse()
    {
        experiment.ReportResultToServer(responseSlider.GetComponent<Slider>().value);
        HideResponseSlider();
    }

    public void AssignActiveExperiment(Experiment exp)
    {
        experiment = exp;
    }

    public void QuitApp()
    {
        Application.Quit();
#if UNITY_EDITOR
        EditorApplication.ExitPlaymode();
#endif
    }

    public void RestartExperiment()
    {
        experiment.Restart();
    }

    public void ReplayTrial()
    {
        experiment.Replay();
    }

    public void ToggleMenu()
    {
        foldoutMenu.SetActive(!foldoutMenu.activeInHierarchy);
        foldoutMenuIcon.transform.Rotate(new Vector3(0f, 0f, 180f));
    }
}
