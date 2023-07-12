## Particle Effect Demo

<video controls muted style="width: 100%;">
  <source src="https://github.com/facebookresearch/aepsych/assets/80999718/f03bd171-fa19-449c-8807-af891cc8629c" type="video/mp4" alt="Particle Effects Demo">
  Your browser does not support the video tag.
</video>

**Description**

This demo showcases a 5-dimensional perceptual optimization, where the optimal condition has a high degree of subjectivity. Each trial, AEPsych generates two new random particle effects by modifying particle hue, initial speed, gravity, glow, and lifetime. The user is then prompted with a subjective pairwise evaluation to indicate which of the two generated particles are more optimal, in this case: “which of the two particles looks more like fire?”. Within about 30 repetitions, AEPsych builds a reasonable Gaussian distribution to predict the parameter combinations of a fire particle effect. This showcases AEPsych’s ability to perform high-dimensional subjective optimizations of arbitrary conditions with relatively sparse data.

**Output**

By adjusting the evaluation condition, AEPsych can produce vastly different outputs. Here’s two examples of AEPsych’s optimal parameter suggestion after optimizing for water vs fire:

<div style="text-align: center;">
  <img src="https://github.com/facebookresearch/aepsych/assets/80999718/083b03c2-a7ed-43ed-9884-ed6665b1bc4f" alt="Water Demo GIF" width="300"/>
  <img src="https://github.com/facebookresearch/aepsych/assets/80999718/253f03e0-a3b1-42b8-a95a-dfb6b139f4dd" alt="Fire Demo GIF" width="300"/>
</div>

**Requirements**

Windows or MacOS, and a local installation of the AEPsych server.

**Installation**

Download the .zip file that corresponds to your operating system
Right click the .zip archive, and “Extract All” to somewhere on your computer
Start your local AEPsych server
Open the extracted archive directory, double click the ‘aepsych_unity.exe’ application

**Download**

[ParticleEffectDemo_Win.zip](https://github.com/facebookresearch/aepsych/files/12034005/ParticleEffectDemo_Win.zip)
