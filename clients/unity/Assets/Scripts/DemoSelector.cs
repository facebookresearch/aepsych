/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

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
