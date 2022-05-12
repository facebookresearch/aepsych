/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AEPsych
{
    public class CSVWriter
    {
        public string filePath;
        bool didWrite = false;

        public CSVWriter(string expName)
        {
            string timestamp = DateTime.Now.ToString("dd-mm-yyyy_hh-mm-ss");
            filePath = string.Format("{0}/{1}", Application.persistentDataPath, expName);
            if (!Directory.Exists(filePath))
            {
                Directory.CreateDirectory(filePath);
            }
            filePath += string.Format("/{0}.csv", timestamp);
        }

        public void StopRecording()
        {
            if (didWrite)
                Debug.Log("Experiment data saved to " + filePath);
        }

        public void WriteTrial(TrialData data)
        {
            if (!File.Exists(filePath))
            {
                using (StreamWriter sw = File.CreateText(filePath))
                {
                    sw.WriteLine(data.GetHeader());
                    sw.WriteLine(data.ToString());
                }
            }
            else
            {
                using (StreamWriter sw = File.AppendText(filePath))
                {
                    sw.WriteLine(data.ToString());
                }
            }
            didWrite = true;
        }

    }
}
