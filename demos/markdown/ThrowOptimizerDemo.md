## VR Throw Optimizer Demo

<video controls muted style="width: 100%;">
  <source src="https://github.com/facebookresearch/aepsych/assets/80999718/a62f7fef-bb0f-4624-891f-7c49164a8e2a" type="video/mp4" alt="Throw Optimizer Demo">
  Your browser does not support the video tag.
</video>

**Description**

The throw optimizer demonstrates AEPsych’s continuous outcome model capabilities. AEPsych's continuous models can find parameters that optimize performance as well as preferences: here, we optimize a controller-based VR throw interaction to maximize throw accuracy.

Phase 1: Throw Physics Continuous Optimization
AEPsych modifies the object offset when held - equating to how the user holds projectile in their hand, release threshold - equating to the time during the throw that the user releases the object, and the number of frames to average over - representing a wrist-flick vs. arm-shot put style trade off. The user gets three tries to throw an object with these modified properties towards the center of a target on the floor, and AEPsych receives the average score across the final two throws as a continuous outcome indicating the quality of those parameters. After 24 trials, users are given their optimal throw parameters for which they are most accurate.

Phase 2: Pairwise Visual Preference Optimization
As an added bonus, users can further tune the visual aesthetic of their throwable object via an AEPsych pairwise optimization in which the user evaluates the subjective properties of parameterized fireball particle effects to tune their visual feedback. This in effect layers performance-tuned physics parameters with preference-tuned visual style to create an individualized interaction system driven entirely by Gaussian Process models.

**Requirements**

Windows OS, Oculus Quest/Quest 2 connected via link cable, and a local AEPsych server installation

**Installation**
1.  Download the .zip file that corresponds to your operating system
2.  Right click the .zip archive, and “Extract All” to somewhere on your computer
3.  Start your local AEPsych server
4.  Launch Quest Link between your Windows machine and your Oculus Quest, and ensure that the headset displays the link home page.
5.  Open the extracted archive directory, double click the ‘Throw Optimizer.exe’ application
6.  Don your headset and follow the on-screen instructions

**Download**
