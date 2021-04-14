using UnityEngine;
using UnityEditor;
using System;

public class DemoSelector : MonoBehaviour
{
    [SerializeField]
    [HideInInspector]
    private int m_Dim = 0;

    [SerializeField]
    [HideInInspector]
    private int m_Method = 0;

    [SerializeField]
    [HideInInspector]
    private int m_Response = 0;

    [SerializeField]
    [HideInInspector]
    private string selectedScript = "None";

}

