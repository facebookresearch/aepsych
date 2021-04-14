using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
public class WriteOutput : MonoBehaviour{

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
    public static string GetDateTime()
    {
        string dateAndTimeVar = System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss");
       // print(dateAndTimeVar);
        return dateAndTimeVar;
    }
    public static void MakeDir(string path)
    {
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
    }
    public static void WriteLine(string path, string line)
    {
     

        //Write some text to the test.txt file
        StreamWriter writer = new StreamWriter(path, true);
        writer.WriteLine(line);
        writer.Close();


    }
}
