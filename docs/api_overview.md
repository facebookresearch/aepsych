---
id: api_overview
title: API Overview
---

The structure of the AEPsych API can be seen in the following diagram, where each component corresponds to a Python class:

![AEPsych API](assets/api_diagram.png)

Below is an overview of what each class does. For more details, see the [API Reference](/api).

- **Sequential Strategy**: The sequential strategy is a container that holds one or more strategies.. Once a strategy finishes (typically by completing a specified number of trials), the next strategy in the sequential strategy begins.

- **Strategy**: Strategies are the building blocks of your experiment. Each strategy has a generator, which determines how AEPsych recommends the next set of points to sample next. A strategy may also optionally make use of a model, depending on its generator. Typically an experiment will start with a strategy that uses a model-less generator, such as the SobolGenerator (explained below). The data from these points are then used to initialize models in subsequent model-based strategies, which sample the parameter space in more intelligent ways.

- **[Generator](/api/aepsych.generators)**: Generators generate the sets of points that AEPsych recommends you sample next. One of the simplest examples of a generator is the SobolGenerator, which samples points in the parameter space quasi-randomly. A more complex example is the OptimizeAcqfGenerator, which makes use of an acquisition function and the strategy's model to sample new points according to some broader goal, such as finding parameter values that maximize the probability of a particular response.

- **[Acquisition Function](/api/aepsych.acquisition)**: Acquisition functions use the strategy's model to determine which points should be sampled next, with some overarching goal in mind. For example, LevelSetEstimation selects the points that will help the model estimate the location of a perceptual threshold (e.g., the set of points where a subject has a 75% of detecting the stimulus). Another example is qNoisyExpectedImprovement, which selects the points that will help the model estimate the maximum of the parameter space (e.g., the parameter values  that maximize the probability a subject detects the stimulus).

- **[Model](/api/aepsych.models)**: The model uses the collected data to compute response probabilities at any point in the parameter space. Note that models can be used outside of strategies to analyze data collected from previous experiments. Currently the only models that are implemented are gaussian process models, which are specified through a mean function and a covariance function (in most cases, you can use AEPsych's defaults). The simplest example of a model is GPClassificationModel, which makes no assumptions other than that responses can be 0 or 1, and responses nearby in the parameter space are correlated. A more complex example is MonotonicRejectionGP, which makes the additional assumption that response probabilities are monotonically increasing along certain dimensions.

You may implement an AEPsych experiment using these classes directly in Python, but users who are not familiar with Python can also configure an AEPsych server using a config file.
