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
        float timer = 0.0f; 
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
