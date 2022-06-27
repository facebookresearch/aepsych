---
id: introduction
title: Introduction
---

AEPsych is a modeling and human-in-the-loop experimentation framework focused on human psychophysics, as well as related domains such as perceptually-informed preferences. It combines state of the art flexible models for the psychometric field, novel active learning objectives targeting psychophysics problems like threshold estimation, and a client-server architecture that supports stimulus presentation using standard packages such as PsychToolbox.

## Why AEPsych (for psychophysics researchers)?

AEPsych shines in high-dimensional psychophysics (as many as 6 or 8 dimensions such as stimulus orientation, size, spatial frequency, etc). In these settings, it can be used to characterize psychometric functions orders of magnitude faster than the method of constant stimuli, and substantially faster than staircase methods as well.

Unlike parametric or heuristic staircase methods, the models in AEPsych make fewer assumptions about the psychometric function, which makes them safer to use in domains where we don’t know if Weber-type scaling laws will hold.

AEPsych always builds a model of the full psychometric field, so even if you focus on threshold estimation you can get some useful signal on JNDs and other properties.

Interactive model exploration and visualization built in.

While the AEPsych modeling is in PyTorch, our innovative client-server architecture means you can write your stimulus presentation code in carefully controlled other environments such as PsychToolbox or PsychoPy, and even run the modeling and stimulus adaptation code on a completely different computer from your presentation computer.

## Why AEPsych (for statistics / ML researchers)

Full API compatibility with BoTorch (LINK) and GPyTorch (LINK) and a full benchmarking suite with novel test functions mean you can evaluate your methods in the ways you are used to while also making it accessible to experimentalists who are not ML experts.

Text-based experiment configuration registry means the same code you use for benchmarking can be used by experimentalists in human-in-the-loop experiments, without them writing any python code.
