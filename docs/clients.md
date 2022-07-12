---
id: clients
title: AEPsych clients
---

The modeling components of AEPsych are by necessity implemented in python, but we recognize that there are popular and well-validated tools for stimulus display in other languages (most notably psychtoolbox in MATLAB), as well as requirements to interface with technologies such as VR that require other languages. To this end, the AEPsych modeling and sample selection algorithms are accessible via a server that clients in different languages can connect to over a network connection (often both client and server run on the same computer).

All clients connect to the server over regular unix sockets, and messages between the client and server are encoded in JSON. Experiments begin with a configuration message from the client to the server, and the proceed with alternating ask messages (which query the server for the next point to evaluate) and tell messages (which let the server know what the outcome of evaluating a stimulus configuration is).

## AEPsych Unity client
The Unity client is written in C#, and supports interfacing AEPsych with stimulus display on regular screens or in VR. It additionally includes tooling for interactive model exploration and model querying, for developing fuller-featured adaptive experiments and prototypes using AEPsych.

## AEPsych MATLAB client
The MATLAB client supports interfacing AEPsych with standard MATLAB stimulus display code (e.g. in psychtoolbox). It has fewer interactive components than the Unity client, but includes core configure-ask-tell functionality for running experiments.

## AEPsych Python client
The AEPsych python client has similar functionality to the MATLAB client, for supporting interfacing with tools such as PsychoPy / OpenSesame. It’s also possible for experiments in python to use the AEPsych machinery directly, not over a network socket, but there are advantages to using the server anyway (e.g. automated data logging and experiment replay functionality, asynchronous operation, etc).
