using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(RectTransform), typeof(Canvas))]
public class DefaultUI : MonoBehaviour
{
    Canvas canvas;
    TextMeshProUGUI experimentText;

    private void Awake()
    {
        experimentText = GetComponentInChildren<TextMeshProUGUI>();
    }

    public void SetText(string msg)
    {
        experimentText.text = msg;
    }
}




