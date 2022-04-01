using AEPsych;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PauseManager : MonoBehaviour
{
    public Experiment experiment;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            if (!experiment.isActiveAndEnabled)
            {
                experiment.gameObject.SetActive(true);
                experiment.SetText("");
            }
            else
            {
                experiment.PauseExperiment();
                experiment.SetText("Paused");
            }

        }
    }
}
