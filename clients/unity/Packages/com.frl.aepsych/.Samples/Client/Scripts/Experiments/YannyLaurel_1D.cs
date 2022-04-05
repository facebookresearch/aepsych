using System.Collections;
using UnityEngine;
using AEPsych;
using TMPro;

public class YannyLaurel_1D : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "YannyLaurel_1D";
    #endregion

    public AudioSource audioSource;

    private void Start()
    {
        if (startMode == StartType.PressAnyKey)
        {
            SetText("Press any key to begin.");
        }
    }

    // ShowStimuli (MANDATORY)
    //      Modify and display a stimulus based on trial parameters.
    //      Use the dimension names defined in your config file as string
    //      keys to access the float lists contained in the TrialConfig.
    //
    // IMPORTANT: Experiment will not continue running until EndShowStimuli is called.
    //
    // EXAMPLE: to read a dimension named "dim1", write:
    //
    //      config["dim1"][0]
    //
    //      The trailing [0] indexer is necessary because the
    //      dict value is a list of floats.
    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        audioSource.pitch = config["pitch"][0];
        audioSource.Play();
        StartCoroutine(EndShowStimuliAfterSeconds(1f));
    }

    // BeginExperiment (optional)
    // Overrides the base BeginExperiment in order to setup the OnStateChange callback
    public override void BeginExperiment()
    {
        onStateChanged += OnExperimentStateChange;
        base.BeginExperiment();
    }

    // PauseExperiment (optional)
    // Overrides base PauseExperiment to remove delegate callback when paused
    public override void PauseExperiment()
    {
        onStateChanged -= OnExperimentStateChange;
        base.PauseExperiment();
    }

    // EndShowStimuliAfterSeconds (optional)
    // helper function, calls EndShowStimuli() after a number of seconds
    IEnumerator EndShowStimuliAfterSeconds(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        SetText("Did you hear Yanny (Y) or Laurel (L)?");
        EndShowStimuli();
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (newState == ExperimentState.WaitingForAskResponse)
        {
            SetText("Querying for next trial");
        }
        AEPsychClient.Log(string.Format("SetState: {0} -> {1}", oldState, newState));
    }

    public override string GetName()
    {
        return experimentName;
    }
}
