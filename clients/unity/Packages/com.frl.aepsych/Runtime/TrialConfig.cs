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
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace AEPsych
{

    [Serializable]
    public class TrialConfig : IDictionary<string, object>
    {
        // Dictionary wrapper for better error messaging if key not found
        Dictionary<string, object> innerDict;
        public TrialConfig()
        {
            innerDict = new Dictionary<string, object>();
        }

        public TrialConfig(Dictionary<string, object> dict)
        {
            innerDict = dict;
        }

        public object this[string key]
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
                    foreach (KeyValuePair<string, object> pair in this)
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

        public List<float> GetFlatList(string key)
        {
                try
                {
                    var keyVal = ((JArray)innerDict[key]).Select(y => y.ToObject<float>()).ToList()
                        .ToList();
                    return keyVal;
                }
                catch (KeyNotFoundException)
                {
                    string allKeys = "";
                    foreach (KeyValuePair<string, object> pair in this)
                    {
                        allKeys += " " + pair.Key + ",";
                    }
                    allKeys = allKeys.Trim(',');
                    throw new KeyNotFoundException($"Dimension name <{key}> is not a valid key in dict. TrialConfig dict keys are:" + allKeys);
                }
        }

        public List<List<float>> GetNestedList(string key)
        {
            try
            {
                var keyVal = ((JArray)innerDict[key]).Select(x => ((JArray)x).Select(y => y.ToObject<float>()).ToList())
                    .ToList();
                return keyVal;
            }
            catch (KeyNotFoundException)
            {
                string allKeys = "";
                foreach (KeyValuePair<string, object> pair in this)
                {
                    allKeys += " " + pair.Key + ",";
                }
                allKeys = allKeys.Trim(',');
                throw new KeyNotFoundException($"Dimension name <{key}> is not a valid key in dict. TrialConfig dict keys are:" + allKeys);
            }
        }

        public ICollection<string> Keys()
        {
            return innerDict.Keys;
        }


        public ICollection<object> Values()
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

        public void Add(string key, object value)
        {
            innerDict.Add(key, value);
        }

        public void Add(KeyValuePair<string, List<object>> item)
        {
            innerDict.Add(item.Key, item.Value);
        }

        public void Add(KeyValuePair<string, object> item)
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
            foreach (KeyValuePair<string, object> pair in innerDict)
            {
                t.Add(pair.Key, pair.Value);
            }
            return t;
        }

        public IEnumerator<KeyValuePair<string, object>> GetEnumerator()
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

            foreach (var pair in innerDict)
            {
                if (pair.Value is List<float>)
                {
                    var values = pair.Value as List<float>;
                    output += "|  " + pair.Key + ": [";
                    foreach (float val in values)
                    {
                        output += val + ",";
                    }
                    output = output.Trim(',');
                    output += "]  |  ";
                }
                else
                {
                    {
                        var values = pair.Value as List<List<float>>;
                        output += "|  " + pair.Key + ": [";
                        foreach(List<float> list in values)
                        {
                            output += "[";
                            foreach (float val in list)
                            {
                                output += val + ","; //watch out for trailing comma ,
                            }

                            output += "]";
                        }
                    }
                }

            }
            return output;
        }


        public bool TryGetValue(string key, out List<float> value)
        {
            bool ret = innerDict.TryGetValue(key, out object flatReturn);
            value = flatReturn as List<float>;
            return ret;
        }

        public bool TryGetValue(string key, out List<List<float>> value)
        {
            bool ret = innerDict.TryGetValue(key, out object flatReturn);
            value = flatReturn as List<List<float>>;
            return ret;
        }

        public bool TryGetValue(string key, out object value)
        {
            return innerDict.TryGetValue(key, out value);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        // UNIMPLEMENTED
        ICollection<string> IDictionary<string, object>.Keys => throw new NotImplementedException();

        ICollection<object> IDictionary<string, object>.Values => throw new NotImplementedException();

        int ICollection<KeyValuePair<string, object>>.Count
        {
            get
            {
                return innerDict.Count;
            }
        }

        bool ICollection<KeyValuePair<string, object>>.IsReadOnly => throw new NotImplementedException();

        public bool Contains(KeyValuePair<string, object> item)
        {
            throw new NotImplementedException();
        }


        public bool Remove(KeyValuePair<string, object> item)
        {
            throw new NotImplementedException();
        }
        public void CopyTo(KeyValuePair<string, object>[] item, int len)
        {
            throw new NotImplementedException();
        }
    }



}
