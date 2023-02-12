---
id: api_overview
title: API Overview
---

The structure of the AEPsych API can be seen in the following diagram, where each component corresponds to a Python class:

![AEPsych API](assets/new_api_diagram.png)


- **[AEPsychStrategy](../aepsych/strategy.py#L500)**: Uses the specified Generation Strategy and AxClient configurations to provide easier functionalties to control, and query the entire experiments. It provides helper functions including but not limited to `plot_contours`, where predictions for a 2-d slice of the parameter space are plotted, and `plot_slice` for 1-d slice parameter space plot.

    - Init Args:
        - `strategy`: The `GenerationStragegy` to use.
        - `ax_client`: the configured `AxClient` to use.


- **[AxClient](https://github.com/facebook/Ax/blob/main/ax/service/ax_client.py#L108)**: generates the next suggestion in the experimentation cycle based on the precised generation strategy, or intelligently chooses the generation strategy if none is provided. Returns the log data from the suggestion's evaluation. Cycle Scheduling is externally handled.

    - Init Args:
        - `generation_strategy`: Optional generation strategy. If not set, one is intelligently chosen based on properties of search space.
        - `db_settings`: Settings for saving and reloading the underlying experiment to a database. Expected to be of type
        - `enforce_sequential_optimization`: Whether to enforce that when it is reasonable to switch models during the optimization (as prescribed
            by `num_trials` in generation strategy), Ax will wait for enough trials to be completed with data to proceed. 
        - `random_seed`: Optional integer random seed, set to fix the optimization
            random seed for reproducibility.
        - `torch_device`: An optional `torch.device` object, used to choose the device used for generating new points for trials.
        - `verbose_logging`: Whether Ax should log significant optimization events, defaults to `True`.
        - `suppress_storage_errors`: Whether to suppress SQL storage-related errors if encounted. 
        - `early_stopping_strategy`: A `BaseEarlyStoppingStrategy` that determines whether a trial should be stopped given the current state of
            the experiment. Used in `should_stop_trials_early`.
        - `global_stopping_strategy`: A `BaseGlobalStoppingStrategy` that determines whether the full optimization should be stopped or not.
    
    -

- **[GenerationStrategy](https://github.com/facebook/Ax/blob/main/ax/modelbridge/generation_strategy.py#L46)**: This describes which generator is to be used for generation at a which point in the trials. It takes as inputs a list of GenerationSteps, each step corresponding to a single generator used, the number of trials that will be generated with this model and the minimum observations required to proceed to the next model. 
    - Init Args:
        - `steps`: A list of `GenerationStep` describing steps of this strategy.
        - `name`: An optional name for this generaiton strategy.
    - Config Args:
        ```
            strategy_names = [init_strat, opt_strat] # The strategies that will be used, corresponding to the named sections below
    
            [init_strat]
            generator = SobolGenerator # Start trial generation with sobol samples.
            min_total_tells = 2
    
            [opt_strat]
            generator = OptimizeAcqfGenerator # after sobol, do model-based active-learning
            min_total_tells = 3
            acqf = qNoisyExpectedImprovement # The acquisition function to be used with the model. We recommend qNoisyExpectedImprovement for optimization problems.
            model = ContinuousRegressionGP # Basic model for continuous outcomes.
        ```

- **[Experiment](https://github.com/facebook/Ax/blob/main/ax/core/experiment.py#L59)**: An Experiment defines the entire set of parameter objects, and parameter constraints of the parameter space, through the `SearchSpace`. An Experiment defines the optimization configuration, which comprises an objective, outcome constraints and an optional risk measure. It also defines the tracking Metrics.

    - Init Args:
        - `search_space`: Search space of the experiment.
        - `name`: Name of the experiment.
        - `optimization_config`: Optimization config of the experiment.
        - `tracking_metrics`: Additional tracking metrics not used for optimization.
        - `runner`: Default runner used for trials on this experiment.
        - `status_quo`: Arm representing existing "control" arm.
        - `description`: Description of the experiment.
        - `is_test`: Convenience metadata tracker for the user to mark test experiments.
        - `experiment_type`: The class of experiments this one belongs to.
        - `properties`: Dictionary of this experiment's properties.
        - `default_data_type`: Enum representing the data type this experiment uses.
    - Config Args:
        ```       
            [common]
            use_ax = True
            
            stimuli_per_trial = 1 # The number of stimuli shown in each trial; currently the Ax backend only supports 1
            outcome_types = [continuous] # The type of response given by the participant; can be [binary] or [continuous].
    
            parnames = [par1, par2, par3] # Names of continuous parameters.
            lb = [0, 0, 1] # Lower bounds of the continuous parameters, in the same order as above.
            ub = [10, 10, 100]
            choice_parnames = [par4, par5]
            fixed_parnames = [par6, par7]
        ```
- **[AxSobolGenerator](../aepsych/generators/sobol_generator.py#L93)**: Generator generates quasi-random low discrepancy eperimentation points for the parameter space.
    - Config Args:
    ```
            [sobol_strat]
            generator = SobolGenerator # points will be generated from Sobol sequence
            min_total_tells = 5 # next strategy will start after at least 5 responses are recorded
    ```
                            


- **[AxOptimizeAcqfGenerator](../aepsych/generators/optimize_acqf_generator.py#L182)**: This is a wrapper for Ax's BOTORCH_MODULAR, consisting of two main components, the AEPsychSurrogate and the AEPsychAcquisitionFunction. It aims to abstract away the interfaces between AEPsych, Ax and Botorch.
    - Config Args:
    ```
            [opt_strat]
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            acqf = qNoisyExpectedImprovement
            min_total_tells = 100
        
            # Configuration for the model
            [ContinuousRegressionGP]
            max_fit_time = 0.1
    ```

- **[AEPsychSurrogate](../aepsych/models/surrogate.py#L18)**: This is wrapper arround BoTorch's Model class, it determines what model to use, its options, its max_fit_time, and its model class. It uses the collected data to compute the response probabilities an any point in the parameter space.

    - Init Args:
        - `max_fit_time`: Optional field that sets an estimated maximum amount of time for the evaluation of all samples
        - `botorch_model_class`: `Model` class to be used as the underlying BoTorch model.
        - `model_options`: Dictionary of options / kwargs for the BoTorch `Model` constructed during `Surrogate.fit`.
        - `mll_class`: `MarginalLogLikelihood` class to use for model-fitting.
        - `mll_options`: Dictionary of options / kwargs for the MLL.
        - `outcome_transform`: BoTorch outcome transforms. Passed down to the BoTorch `Model`. Multiple outcome transforms can be chained
            together using `ChainedOutcomeTransform`.
        - `input_transform`: BoTorch input transforms. Passed down to the BoTorch `Model`. Multiple input transforms can be chained
            together using `ChainedInputTransform`.
        <!-- - `covar_module_class`: Covariance module class, not yet used. Will be used to construct custom BoTorch `Model` in the future.
        - `covar_module_options`: Covariance module kwargs, not yet used. Will be used to construct custom BoTorch `Model` in the future.
        - `likelihood`: `Likelihood` class, not yet used. Will be used to construct custom BoTorch `Model` in the future.
        - `likelihood_options`: Likelihood options, not yet used. Will be used to construct custom BoTorch `Model` in the future. -->

    - Config Args:
    ```
            [opt_strat]
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
    ```

- **[AEPsychAcquisition](../aepsych/acquisition/acquisition.py#L15)**: Acquisition functions use the strategy's model to determine which points should be sampled next, with some overarching goal in mind. We recommend PairwiseMCPosteriorVariance for global exploration, and qNoisyExpectedImprovement for optimization. For other options, check out the botorch and aepsych docs.
    - Config Args:
    ```
            [opt_strat]
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            acqf = qNoisyExpectedImprovement
    ```

You may implement an AEPsych experiment using these classes directly in Python, but users who are not familiar with Python can also configure an AEPsych server using a config file.
