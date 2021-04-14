using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FlashSprite : MonoBehaviour
{
    public float flashDuration = 1.5f;
    public float alpha;
    public float r; public float g; public float b;
   
    // Start is called before the first frame update
    void Start()
    {
        Color c = GetComponent<SpriteRenderer>().color;
        GetComponent<SpriteRenderer>().color = new Color(r, g, b, alpha);
        //destory after time
        Destroy(this.gameObject, flashDuration);
    }
    public void SetGrayscaleColor(float f, float a=1.0f)
    {
        r = f; g = f; b = f; alpha = a;
        GetComponent<SpriteRenderer>().color = new Color(r,g,b, alpha);
    }
    public void SetColor(float r, float g, float b, float a = 1.0f)
    {
        this.r = r; this.g = g; this.b = b; this.alpha = a;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
