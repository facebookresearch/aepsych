using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(RectTransform), typeof(Canvas))]
public class DefaultUI : MonoBehaviour
{
    public TextMeshProUGUI experimentText;

    public void SetText(string msg)
    {
        if (experimentText != null)
            experimentText.text = msg;
    }

    public void SetTextActive(bool active)
    {
        experimentText.enabled = active;
    }
}




