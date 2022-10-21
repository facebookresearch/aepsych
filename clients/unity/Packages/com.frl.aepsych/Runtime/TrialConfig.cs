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
using UnityEngine;

namespace AEPsych
{
    [Serializable]
    public class TrialConfig : IDictionary<string, List<float>>
    {
        // Dictionary wrapper for better error messaging if key not found
        Dictionary<string, List<float>> innerDict;

        public TrialConfig()
        {
            innerDict = new Dictionary<string, List<float>>();
        }

        public TrialConfig(Dictionary<string, List<float>> dict)
        {
            innerDict = dict;
        }

        public List<float> this[string key]
        {
            get
            {
                try
                {
                    return innerDict[key];
                }
                catch (KeyNotFoundException)
                {
                    string allKeys = "";
                    foreach (KeyValuePair<string, List<float>> pair in this)
                    {
                        allKeys += " " + pair.Key + ",";
                    }
                    allKeys = allKeys.Trim(',');
                    throw new KeyNotFoundException($"Dimension name <{key}> is not a valid key in dict. TrialConfig dict keys are:" + allKeys);
                }
            }
            set
            {
                innerDict[key] = value;
            }
        }

        public ICollection<string> Keys()
        {
            return innerDict.Keys;
        }

        public ICollection<List<float>> Values()
        {
            return innerDict.Values;
        }

        public int Count()
        {
            return innerDict.Count;
        }
        public bool IsReadOnly()
        {
            return false;
        }

        public void Add(string key, List<float> value)
        {
            innerDict.Add(key, value);
        }

        public void Add(KeyValuePair<string, List<float>> item)
        {
            innerDict.Add(item.Key, item.Value);
        }

        public void Clear()
        {
            innerDict.Clear();
        }

        public bool Contains(KeyValuePair<string, List<float>> item)
        {
            if (!innerDict.ContainsKey(item.Key))
                return false;
            if (!innerDict.ContainsValue(item.Value))
                return false;
            return true;
        }

        public bool ContainsKey(string key)
        {
            return innerDict.ContainsKey(key);
        }

        public TrialConfig Copy()
        {
            TrialConfig t = new TrialConfig();
            foreach (KeyValuePair<string, List<float>> pair in innerDict)
            {
                t.Add(pair.Key, pair.Value);
            }
            return t;
        }

        public void CopyTo(KeyValuePair<string, List<float>>[] array, int arrayIndex)
        {
            int idx = arrayIndex;
            foreach (KeyValuePair<string, List<float>> pair in innerDict)
            {
                array[idx++] = pair;
            }
        }

        public IEnumerator<KeyValuePair<string, List<float>>> GetEnumerator()
        {
            return innerDict.GetEnumerator();
        }

        public bool Remove(string key)
        {
            return innerDict.Remove(key);
        }

        public bool Remove(KeyValuePair<string, List<float>> item)
        {
            return innerDict.Remove(item.Key);
        }

        public override string ToString()
        {
            string output = "";
            foreach (KeyValuePair<string, List<float>> pair in innerDict)
            {
                output += "|  " + pair.Key + ": [";
                foreach (float val in pair.Value)
                {
                    output += val + ",";
                }
                output = output.Trim(',');
                output += "]  |  ";
            }
            return output;
        }

        public bool TryGetValue(string key, out List<float> value)
        {
            return innerDict.TryGetValue(key, out value);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        // UNIMPLEMENTED
        ICollection<string> IDictionary<string, List<float>>.Keys => throw new NotImplementedException();

        ICollection<List<float>> IDictionary<string, List<float>>.Values => throw new NotImplementedException();

        int ICollection<KeyValuePair<string, List<float>>>.Count
        {
            get
            {
                return innerDict.Count;
            }
        }

        bool ICollection<KeyValuePair<string, List<float>>>.IsReadOnly => throw new NotImplementedException();
    }
}
