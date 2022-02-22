using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AEPsych
{
    //TrialMetadata class can be modified or inherited from to contain any trial info.
    //It can be passed into Tell messages, and metadata later read from the database.
    //For easiest parsing, variables should be primitive data types not requiring
    //additional serialization (e.g. string, int, float, bool)
    public class TrialMetadata
    {
        public float responseTime;
        public string participantID;

        public TrialMetadata(float responseTime, string participantId = "")
        {
            this.responseTime = responseTime;
            this.participantID = participantId;
        }
    }
}
