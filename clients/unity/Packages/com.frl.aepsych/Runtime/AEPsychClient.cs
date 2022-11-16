/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace AEPsych
{
    public enum RequestType { setup, ask, tell, resume, query, parameters, can_model, exit };
    public enum QueryType { min, max, prediction, inverse }


    //_______________________________________________________________________
    // AEPsych custom data types
    #region

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
        public Request(object message, RequestType type, TrialMetadata extra_info = null)
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
        public float outcome;

        public TrialWithOutcome(TrialConfig config, float outcome, float response_time = -1f)
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
    public class TrialData
    {
        public string timeStamp;
        public TrialConfig config;
        public TrialMetadata extra_info;
        public float outcome;

        public TrialData(string timeStamp, TrialConfig config, float outcome, TrialMetadata extra_info = null)
        {
            this.timeStamp = timeStamp;
            this.config = config;
            this.outcome = outcome;
            this.extra_info = extra_info;
        }

        public override string ToString()
        {
            string data = timeStamp + ",";
            foreach (KeyValuePair<string, List<float>> pair in config)
            {
                foreach (float x in pair.Value)
                {
                    data += x + ",";
                }
            }
            data += outcome + ",";
            if (extra_info != null)
            {
                data +=  JsonConvert.SerializeObject(extra_info);
            }
            return data;
        }

        public string GetHeader()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Timestamp:,");
            foreach (KeyValuePair<string, List<float>> pair in config)
            {
                sb.Append( pair.Key + ":");
                for (int i = 0; i < pair.Value.Count; i++)
                {
                    sb.Append(",");
                }
            }
            sb.Append("Outcome:,Metadata:");
            return sb.ToString();
        }
    }


    #endregion
    //_______________________________________________________________________


    public class AEPsychClient : MonoBehaviour
    {
        //_______________________________________________________________________
        // AEPsych Fields and Refrences
        #region
        public enum ClientStatus { Ready, QuerySent, GotResponse, FailedToConnect };
        ClientStatus status;
        string reply;
        List<string> serverMessageQueue = new List<string>();
        public TrialConfig baseConfig;
        public int currentStrat;
        int retryConnectCount = 0;
        int maxRetries = 3;
        public string server_address = "localhost";
        public int server_port = 5555;
        [HideInInspector] public bool finished;
        public bool debugEnabled;

        public delegate void StatusChangeCallback(ClientStatus oldStatus, ClientStatus newStatus);
        public event StatusChangeCallback onStatusChanged;

        private volatile TcpClient tcpConnection;
        private Thread ListenThread;
        private volatile bool connected;
        private volatile bool serverConnectionLost = false;
        #endregion
        //_______________________________________________________________________


        //_______________________________________________________________________
        // Unity Built-in Methods
        #region
        private void Start()
        {
            Application.runInBackground = true;
        }

        private void Update()
        {
            if (serverMessageQueue.Count > 0)
            {
                while (serverMessageQueue.Count > 0)
                {
                    reply += serverMessageQueue[0];
                    serverMessageQueue.RemoveAt(0);
                }
            }
            if (serverConnectionLost)
            {
                SetStatus(ClientStatus.FailedToConnect);
                serverConnectionLost = false;
            }
        }

        void OnApplicationQuit()
        {
            CloseServer();
        }

        private void OnValidate() => PlayerPrefs.SetInt("AEPsychDebugEnabled", debugEnabled ? 0 : 1);

#endregion
        //_______________________________________________________________________


        //_______________________________________________________________________
        // Public Methods
#region
        public IEnumerator StartServer(string configPath, string version = "0.01", bool isPath = true)
        {
            finished = false;
            reply = null;
            if (tcpConnection == null)
            {
                ConnectServer();
            }
            yield return new WaitUntil(() => tcpConnection != null);
            string configStr;
            if (isPath)
            {
                configStr = ReadFile(configPath);
            }
            else
            {
                configStr = configPath;
            }
            SetStatus(ClientStatus.QuerySent);
            SetupMessage setupMessage = new SetupMessage(configStr: configStr);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(new VersionedRequest(setupMessage, RequestType.setup, version))));
        }

        public void ConnectServer()
        {
            try
            {
                ListenThread = new Thread(new ThreadStart(ConnectToServer));
                ListenThread.IsBackground = true;
                ListenThread.Start();
            }
            catch (Exception e)
            {
                Debug.Log("On client connect exception " + e);
            }
            SetStatus(ClientStatus.Ready);
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
            SetStatus(ClientStatus.Ready);
            if (version == "0.01")
            {
                try
                {
                    TrialWithFinished config = JsonConvert.DeserializeObject<TrialWithFinished>(reply);
                    finished = config.is_finished;
                    baseConfig = config.config;
                }
                catch
                {
                    Debug.LogError("Tried to deserialize invalid server message: " + reply);
                }
            }
            else
            {
                try
                {
                    baseConfig = JsonConvert.DeserializeObject<TrialConfig>(reply);
                }
                catch
                {
                    Debug.LogError("Tried to deserialize invalid server message: " + reply);
                }
            }

            return baseConfig;
        }

        public int GetStrat() //can only call this after setup
        {
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called getConfig() when there is no reply available! Current status is " + status);
            }
            SetStatus(ClientStatus.Ready);
            Debug.Log(reply);
            currentStrat = JsonConvert.DeserializeObject<int>(reply);
            return currentStrat;
        }

        public IEnumerator Tell(TrialConfig trialConfig, float outcome, TrialMetadata metadata = null, float responseTime = -1f)
        {
            TrialWithOutcome message = new TrialWithOutcome(trialConfig, outcome, responseTime);
            Request req;
            //req = new Request(message, RequestType.tell, metadata, responseTime);
            req = new Request(message, RequestType.tell, metadata);
            
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
            QueryMessage queryResponse = null;
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called getQuery() when there is no reply available! Current status is " + status);
            }
            SetStatus(ClientStatus.Ready);
            try
            {
                queryResponse = JsonConvert.DeserializeObject<QueryMessage>(reply);
            }
            catch
            {
                Debug.LogError("Failed to deserialize invalid server query response: " + reply);
            }
            return queryResponse;
        }

        public IEnumerator Resume(int strat_id, string version = "0.01")
        {
            finished = false;
            currentStrat = strat_id;
            ResumeMessage message = new ResumeMessage(strat_id);
            VersionedRequest req = new VersionedRequest(message, RequestType.resume, version);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }

        public IEnumerator CheckForModel()
        {
            Request req = new Request("", RequestType.can_model);
            yield return StartCoroutine(this.SendRequest(JsonConvert.SerializeObject(req)));
        }

        public bool GetModelResponse()
        {
            if (status != ClientStatus.GotResponse)
            {
                Debug.Log("Error! Called GetModelResponse() when there is no reply available! Current status is " + status);
            }
            SetStatus(ClientStatus.Ready);
            Dictionary<string, bool> modelReply = JsonConvert.DeserializeObject<Dictionary<string, bool>>(reply);
            return modelReply["can_model"];
        }

        public void CloseServer()
        {
            Log("Closing connection.");
            connected = false;
            Request req = new Request("", RequestType.exit);
            SendMessageToServer(JsonConvert.SerializeObject(req));
        }

        public static void Log(string msg)
        {
            if (PlayerPrefs.GetInt("AEPsychDebugEnabled") == 0)
            {
                Debug.Log(msg);
            }
        }
#endregion
        //_______________________________________________________________________


        //_______________________________________________________________________
        // Internal Methods
#region
        public string ReadFile(string filePath)
        {
            var sr = new StreamReader(filePath);
            string fileContents = sr.ReadToEnd();
            sr.Close();
            return fileContents;
        }

        public IEnumerator SendRequest(string query)
        {
            reply = null;
            SendMessageToServer(query);
            SetStatus(ClientStatus.QuerySent);
            yield return new WaitUntil(() => reply != null);
            SetStatus(ClientStatus.GotResponse);
            yield return null;
        }

        public void SetStatus(ClientStatus newStatus)
        {
            ClientStatus oldStatus = status;
            if (newStatus != oldStatus)
            {
                status = newStatus;
                if (onStatusChanged != null)
                {
                    onStatusChanged(oldStatus, newStatus);
                }
            }
        }

        public void LogFromInstance(string msg)
        {
            if (debugEnabled)
            {
                Debug.Log(msg);
            }
        }

        #endregion
        //_______________________________________________________________________


        //_______________________________________________________________________
        // Server Communication Methods
        #region
        public void ConnectToServer()
        {
            if (tcpConnection != null)
            {
                return;
            }

            try
            {
                // Create a TcpClient.
                LogFromInstance(string.Format("Attempting connection to server at {0}:{1}", server_address, server_port));
                tcpConnection = new TcpClient(server_address, server_port);
                LogFromInstance(tcpConnection.Connected ? "Successfully connected to server." : "Failed to connect to server.");
                connected = tcpConnection.Connected;
                Byte[] data = new Byte[1024];
                while (connected)
                {
                    // Get a stream object for reading
                    using (NetworkStream stream = tcpConnection.GetStream())
                    {
                        int length;
                        // Read incomming stream into byte arrary.
                        while (connected && stream != null && (length = stream.Read(data, 0, data.Length)) != 0)
                        {
                            Byte[] serverData = new byte[length];
                            Array.Copy(data, 0, serverData, 0, length);
                            // Convert byte array to string message.
                            string serverMessage = Encoding.ASCII.GetString(serverData);
                            LogFromInstance("Received: " + serverMessage);
                            serverMessageQueue.Add(serverMessage);
                        }
                    }
                }
                tcpConnection.Close();
            }
            //catch (SocketException e)
            catch (Exception e)
            {
                if (e.GetType() == typeof(ThreadAbortException))
                {
                    LogFromInstance("Aborting client listen thread.");
                }
                else if (e.GetType() == typeof(SocketException))
                {
                    if (retryConnectCount++ < maxRetries)
                    {
                        Debug.Log("connection timed out. Retrying. Retry count: " + retryConnectCount);
                        ConnectToServer();
                    }
                    else
                    {
                        serverConnectionLost = true;
                        ServerFailure(string.Format("AEPsych Server not found at {0}:{1}. Ensure that the server has been installed and initialized.", server_address, server_port));
                    }
                }
                else
                {
                    ServerFailure(String.Format("Socket exception: {0}: {1}", e.GetType(), e.Message));
                    serverConnectionLost = true;
                }

            }
        }

        public void SendMessageToServer(string msg)
        {
            if (tcpConnection == null || !tcpConnection.Connected)
            {
                Debug.LogError("Message Send failed. Socket is closed.");
                return;
            }
            try
            {
                // Ignore empty messages
                if (msg.Length == 0)
                {
                    Log("sending empty message");
                    msg = "";
                }
                // Translate the passed message into ASCII and store it as a Byte array.
                Byte[] data = Encoding.ASCII.GetBytes(msg);

                // Get a client stream for reading and writing.
                NetworkStream stream = tcpConnection.GetStream();

                // Send the message to the connected TcpServer.
                stream.Write(data, 0, data.Length);

                Log(string.Format("Sent: {0}", msg));
            }
            catch (SocketException socketException)
            {
                Debug.Log("Socket exception: " + socketException);
            }
        }

        public virtual void ServerFailure(string msg)
        {
            Debug.LogError(msg);
        }
#endregion
        //_______________________________________________________________________
    }
}
