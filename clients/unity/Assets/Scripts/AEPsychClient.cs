/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/


using NetMQ;
using NetMQ.Sockets;
using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.IO;

namespace AEPsych
{
    public enum RequestType { setup, ask, tell, resume, query, parameters };
    public enum QueryType { min, max, prediction, inverse }

    // TODO make this a little more strongly typed
    public class TrialConfig : Dictionary<string, List<float>> { }

    public class Request
    {
        // this should definitely be more narrowly defined
        public object message;

        [JsonConverter(typeof(StringEnumConverter))]
        public RequestType type;

        public TrialMetadata extra_info;

        public Request(object message, RequestType type)
        {
            this.message = message;
            this.type = type;
            this.extra_info = null;
        }
        public Request(object message, RequestType type, TrialMetadata extra_info)
        {
            this.message = message;
            this.type = type;
            this.extra_info = extra_info;
        }
    }

    public class VersionedRequest : Request
    {
        public string version;
        public VersionedRequest(object message, RequestType type, string version) : base(message, type)
        {
            this.version = version;
        }
    }

    public class TrialWithFinished
    {
        public TrialConfig config;
        public bool is_finished;

        public TrialWithFinished(TrialConfig config, bool is_finished)
        {
            this.config = config;
            this.is_finished = is_finished;
        }
    }

    public class TrialWithOutcome
    {
        public TrialConfig config;
        public int outcome;

        public TrialWithOutcome(TrialConfig config, int outcome)
        {
            this.config = config;
            this.outcome = outcome;
        }

    }

    public class QueryMessage
    {
        [JsonConverter(typeof(StringEnumConverter))]
        public QueryType query_type;
        public TrialConfig x; //values where we want to query
        public float y; //target that we want to inverse querying
        public TrialConfig constraints; //Constraints for inverse querying; if values are 1d then absolute constraint, if 2d then upper/lower bounds
        public bool probability_space;  //whether to use probability space or latent space

        public QueryMessage(QueryType queryType, TrialConfig x, float y, TrialConfig constraints, bool probability_space)
        {
            this.query_type = queryType;
            this.x = x;
            this.y = y;
            this.constraints = constraints;
            this.probability_space = probability_space;
        }
    }

    public class ResumeMessage
    {
        public int strat_id;

        public ResumeMessage(int strat_id)
        {
            this.strat_id = strat_id;
        }
    }

    public class Target
    {
        public string target;
        public Target(string target)
        {
            this.target = target;
        }
    }


    public class SetupMessage
    {
        public string config_str;

        public SetupMessage(string configStr)
        {
            this.config_str = configStr;
        }
    }


    public class AEPsychClient : MonoBehaviour
    {
        RequestSocket client;
        public enum ClientStatus { Ready, QuerySent, GotResponse };
        ClientStatus status;
        string reply;
        public TrialConfig baseConfig;
        public int currentStrat;
        public string server_address = "tcp://localhost";
        public string server_port = "5555";
        public bool finished;

        public string ReadFile(string filePath)
        {
            var sr = new StreamReader(filePath);
            string fileContents = sr.ReadToEnd();
            sr.Close();
            return fileContents;
        }

        public IEnumerator StartServer(string configPath, string version = "0.01")
        {
            CleanupClient();
            string configStr = ReadFile(configPath);
            AsyncIO.ForceDotNet.Force();
            status = ClientStatus.QuerySent;
            client = new RequestSocket();
            client.Connect($"{this.server_address}:{this.server_port}");
            SetupMessage setupMessage = new SetupMessage(configStr: configStr);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(new VersionedRequest(setupMessage, RequestType.setup, version))));
        }

        public void ConnectServer()
        {
            CleanupClient();
            AsyncIO.ForceDotNet.Force();
            status = ClientStatus.Ready;
            client = new RequestSocket();
            client.Connect($"{this.server_address}:{this.server_port}");
        }


        IEnumerator SendRequest(string query)
        {

            reply = null;
            bool gotMessage = false;
            Debug.Log("Sending " + query);
            client.SendFrame(query);
            status = ClientStatus.QuerySent;
            while (!gotMessage)
            {
                gotMessage = client.TryReceiveFrameString(out reply); // this returns true if it's successful
                yield return null;
            }
            if (gotMessage)
            {
                Debug.Log("Received " + reply);
            }
            status = ClientStatus.GotResponse;
        }

        public ClientStatus GetStatus()
        {
            return status;
        }

        public bool IsBusy()
        {
            return (status == ClientStatus.QuerySent);
        }

        public IEnumerator WaitForReady()
        {
            while (IsBusy())
            {
                yield return null;
            }
        }

        public TrialConfig GetConfig(string version = "0.01")
        {
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called getConfig() when there is no reply available! Current status is " + status);
            }
            status = ClientStatus.Ready;
            if (version == "0.01")
            {
                TrialWithFinished config = JsonConvert.DeserializeObject<TrialWithFinished>(reply);
                finished = config.is_finished;
                baseConfig = config.config;
            }
            else
            {
                baseConfig = JsonConvert.DeserializeObject<TrialConfig>(reply);
            }

            return baseConfig;
        }

        public int GetStrat() //can only call this after setup
        {
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called getConfig() when there is no reply available! Current status is " + status);
            }
            status = ClientStatus.Ready;
            currentStrat = JsonConvert.DeserializeObject<int>(reply);
            return currentStrat;
        }

        public IEnumerator Tell(TrialConfig trialConfig, int outcome, TrialMetadata metadata = null)
        {
            TrialWithOutcome message = new TrialWithOutcome(trialConfig, outcome);
            Request req;
            if (metadata != null)
            {
                req = new Request(message, RequestType.tell, metadata);
            }
            else
            {
                req = new Request(message, RequestType.tell);
            }
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }
        public IEnumerator Ask()
        {
            VersionedRequest req = new VersionedRequest("", RequestType.ask, version: "0.01");
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }

        public IEnumerator Params()
        {
            Request req = new Request("", RequestType.parameters);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }

        public IEnumerator Query(QueryType queryType, TrialConfig x = null, float y = 0, TrialConfig constraints = null, bool probability_space = false)
        {
            if (x == null)
            {
                x = new TrialConfig { };
            }
            if (constraints == null)
            {
                constraints = new TrialConfig() { };
            }

            QueryMessage message = new QueryMessage(queryType, x, y, constraints, probability_space);
            Request req = new Request(message, RequestType.query);
            string s = JsonConvert.SerializeObject(req);
            yield return StartCoroutine(this.SendRequest(s));
        }
        public QueryMessage GetQueryResponse()
        {
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called getQuery() when there is no reply available! Current status is " + status);
            }
            status = ClientStatus.Ready;
            QueryMessage queryResponse = JsonConvert.DeserializeObject<QueryMessage>(reply);
            return queryResponse;
        }

        public IEnumerator Resume(int strat_id, string version = "0.01")
        {
            ResumeMessage message = new ResumeMessage(strat_id);
            VersionedRequest req = new VersionedRequest(message, RequestType.resume, version);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }

        void CleanupClient()
        {
            if (client != null)
            {
                client.Close();
                NetMQConfig.Cleanup();
            }
        }

        void OnApplicationQuit()
        {
            CleanupClient();

        }

    }


}
