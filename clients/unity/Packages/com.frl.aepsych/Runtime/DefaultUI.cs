/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
