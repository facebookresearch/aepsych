using System.Collections;
using UnityEngine;
using AEPsych;
using TMPro;

public class Ex2DSingleDetect : Experiment
{
    // Do Not Modify this region
    #region
    [HideInInspector] public string experimentName = "Ex2DSingleDetect";
    #endregion
    public GameObject circlePrefab;
    public TextMeshProUGUI trialText;
    public string configName = "configs/single_lse_2d.ini";

    // ShowStimuli (MANDATORY)
    // Display a stimulus, and finish when the stimulus is done.
    // Use the dimension names defined in your config file as string keys to access the float lists
    // contained in the TrialConfig. Experiment will not continue running until EndShowStimuli is called.
    public override void ShowStimuli(TrialConfig config)
    {
        SetText("Now presenting stimulus.");
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(config["gsColor"][0], config["alpha"][0]);
        StartCoroutine(EndShowStimuliAfterSeconds(fs.flashDuration));
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
        EndShowStimuli();
        SetText("Was it visible? (Y/N)");
    }

    public override string GetName()
    {
        return experimentName;
    }

    // OnStateChange (optional)
    // An optional callback for when the ExperimentState field changes. This enables
    // additional experiment flow control and visibility into the Experiment State Machine.
    public void OnExperimentStateChange(ExperimentState oldState, ExperimentState newState)
    {
        if (oldState == ExperimentState.WaitingForTellResponse)
        {
            SetText("Querying for next trial");
        }
    }

    private void Start()
    {
        SetText("Welcome. Press Y to begin.");
        StartCoroutine(WaitForInput());
    }

    IEnumerator WaitForInput()
    {
        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Y));
        BeginExperiment();
    }

    void SetText(string s)
    {
        trialText.SetText(s);
    }

    public override void ExperimentComplete()
    {
        StartCoroutine(DisplayOptimal());
        base.ExperimentComplete();
    }

    IEnumerator DisplayOptimal()
    {
        yield return StartCoroutine(client.Query(QueryType.inverse, y: 0.75f, probability_space: true));
        QueryMessage m = client.GetQueryResponse();
        TrialConfig maxLoc = m.x;
        GameObject circle = Instantiate(circlePrefab);
        FlashSprite fs = circle.GetComponent<FlashSprite>();
        fs.SetGrayscaleColor(maxLoc["gsColor"][0], maxLoc["alpha"][0]);
        fs.flashDuration = -1.0f; //never destroy
        SetText("Experiment complete! Displaying optimal color: ");
    }
}
