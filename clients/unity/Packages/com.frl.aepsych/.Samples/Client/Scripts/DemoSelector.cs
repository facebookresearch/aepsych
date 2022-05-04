/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using UnityEngine;
using System;

public class DemoSelector : MonoBehaviour
{
#pragma warning disable CS0414
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
#pragma warning restore CS0414
}
