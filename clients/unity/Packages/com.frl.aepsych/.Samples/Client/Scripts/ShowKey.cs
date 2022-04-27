/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using UnityEngine;

public class ShowKey : MonoBehaviour
{
    Text t;
    float timer;
    // Start is called before the first frame update
    void Start()
    {
        t = GetComponent<Text>();
#pragma warning disable CS0219 // Variable is assigned but its value is never used
        float timer = 0.0f;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
    }

    // Update is called once per frame
    void Update()
    {
        bool keyPress = true;
        if (Input.GetKey(KeyCode.Y))
        {
            t.text = "Y";
        }
        else if (Input.GetKey(KeyCode.N))
        {
            t.text = "N";
        }
        else if (Input.GetKey(KeyCode.Alpha1))
        {
            t.text = "1";
        }
        else if (Input.GetKey(KeyCode.Alpha2))
        {
            t.text = "2";
        }
        else
        {
            keyPress = false;
            timer = timer - Time.deltaTime;
            if (timer < 0)
            {
                t.text = "";
            }
        }
        if (keyPress)
        {
            timer = 0.25f;
        }


    }
}
