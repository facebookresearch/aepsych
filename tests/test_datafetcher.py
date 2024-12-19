from unittest.mock import MagicMock
from .server.test_server import BaseServerTestCase
from aepsych.database.data_fetcher import DataFetcher
from aepsych.config import Config
from typing import Iterable

class DataFetcherTestCase(BaseServerTestCase):
    def default_config(self, outcome_names = ["outcome"], 
                        outcome_types = ["binary"],
                        par_data = {
                           "par1": ["continuous", 0, 1, False],
                           "par2": ["continuous", 0, 1, False]
                        },
                        model_name = "GPClassificationModel",
                        num_stim = 1):
        
        def print_par_data(data):
            par_string = ""
            for k, v in data.items():
                par_string += f"""
                                [{k}]
                                par_type = {v[0]}
                                lower_bound = {v[1]}
                                upper_bound = {v[2]}
                                log_scale = {v[3] if len(v) == 4 else False}
                                """
                
            return par_string
        
        return f"""[metadata]
                  experiment_name = my_exp
                  participant_id = pid3

                  [common]
                  parnames = [{', '.join(name for name in par_data)}]
                  stimuli_per_trial = {num_stim}
                  outcome_names = [{', '.join(name for name in outcome_names)}]
                  outcome_types = [{', '.join(type for type in outcome_types)}]
                  strategy_names = [my_strat]

                  {print_par_data(par_data)}

                  [my_strat]
                  min_asks = 5
                  generator = SobolGenerator
                  model = {model_name}
               """

    def pre_seed_config(self, conds_name, exp_names = None, exp_ids = None, exp_desc = None, par_ids = None, ex_data = None):
        config_str = f"""seed_data_conditions = {conds_name}

                  [{conds_name}]"""
        
        if exp_names:
            config_str += f"\nexperiment_name = [{', '.join(name for name in exp_names)}]"

        if exp_ids:
            config_str += f"\nexperiment_id = [{', '.join(exp_id for exp_id in exp_ids)}]"

        if exp_desc:
            config_str += f"\nexperiment_description = [{', '.join(desc for desc in exp_desc)}]"

        if par_ids:
            config_str +=  f"\nparticipant_id = [{', '.join(par_id for par_id in par_ids)}]"

        if ex_data:
            config_str += "\n" + '\n'.join(f"{k} = {v}" for k, v in ex_data.items())

        return config_str

    @property
    def database_path(self):
        return "./aepsych/tests/test_databases/1000_outcome.db"

    def setUp(self):
        super().setUp()
        setup_message = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": self.default_config()}
        }

        self.s.db.record_config = MagicMock()
        self.s.db.record_message = MagicMock()
        self.s.db.record_setup = MagicMock()

        self.s.handle_request(setup_message)

    def tearDown(self):
        self.s.cleanup()

    def test_create_from_config(self):
        test_names = ["my experiment", "my_exp"]
        test_exp_ids = ["exp1", "exp2", "exp3"]
        test_desc = ["desc1", "desc2"]
        test_par_ids = ["d6e6afba-8a38-49c8-82f8-6b58846b2b34", "bb9c9c60-45cd-4102-aead-30888cc35d9f"]
        test_ex_data = {"some data": "[[1, 2], 2]", "another thing": "bb9c9c60-45cd-4102-aead-30888cc35d9f"}

        config_str = self.default_config() + self.pre_seed_config(
            "seed_conds", 
            test_names,
            test_exp_ids,
            test_desc,
            test_par_ids,
            test_ex_data
        )
        
        test_fetcher = DataFetcher.from_config(Config(config_str= config_str), "my_strat")
        self.assertTrue(
            test_fetcher.experiment_names is not None and 
            len(test_fetcher.experiment_names) == len(test_names) and
            x == test_names[i] for i, x in enumerate(test_fetcher.experiment_names)
        )
        
        self.assertTrue(
            test_fetcher.experiment_ids is not None and
            len(test_fetcher.experiment_ids) == len(test_exp_ids) and
            x == test_exp_ids[i] for i, x in enumerate(test_fetcher.experiment_ids)
        )

        self.assertTrue(
            test_fetcher.experiment_desc is not None and
            len(test_fetcher.experiment_desc) == len(test_desc) and
            x == test_desc[i] for i, x in enumerate(test_fetcher.experiment_desc)
        )

        self.assertTrue(
            test_fetcher.participant_ids is not None and
            len(test_fetcher.participant_ids) == len(test_par_ids) and 
            x == test_par_ids[i] for i, x in enumerate(test_fetcher.participant_ids)
        )
        
        self.assertTrue(
            test_fetcher.extra_metadata is not None and
            len(test_fetcher.extra_metadata) == len(test_ex_data) and
            (k in test_ex_data and test_ex_data[k] == v for k, v in test_fetcher.extra_metadata.items())
        )

    def test_experiment_name_match(self):
        # the ids associated with the multi param experiment
        validation_set = set([1, 7, 9, 15, 18, 20, 27])

        # should filter out pairwise on # of stim
        # should filter out negative on outcome type
        config_str = self.default_config() + self.pre_seed_config(
            "seed_conds", 
            #["my experiment", "my_exp"]
            ["multi_param_gen", "pairwise_opt_gen", "negative_param_exp"],
        )
        
        test_fetcher = DataFetcher.from_config(Config(config_str= config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)
        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))

    def test_experiment_desc_match(self):
        # the ids associated with the multi param experiment
        validation_set = set([1, 7, 9, 15, 18, 20, 27])

        # should filter out pairwise and negative on same basis 
        config_str = self.default_config() + self.pre_seed_config(
            "seed_conds", 
            exp_desc = [
                "\"generating data with multiple params for tests\"",
                "\"generating pairwise data for tests\"",
                "\"generating data with negative param vals for tests\""
            ])
        
        test_fetcher = DataFetcher.from_config(Config(config_str= config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)
        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))

    def test_experiment_id_match(self):
        # the ids associated with the multi param experiment
        validation_set = set([1, 7, 9, 15, 18, 20, 27])

        # though this will match every test in the db it should filter down to only 
        # the multi param experiment because of the config fitlering
        config_str = self.default_config() + self.pre_seed_config(
            "seed_conds", 
            exp_ids = ["f18d43aa-e4e0-4376-859b-8ef8e9d749dd"]
        )
        
        test_fetcher = DataFetcher.from_config(Config(config_str= config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)
        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))

    def test_participant_id_match(self):
        # only the pairwise experiments should meet all criteria for this config 
        # so even though the provided participant ids are associated with other tests
        # they should only meet this subset of the pairwise experiments. 
        validation_set = set([6, 8, 16, 17])
        config_str = self.default_config(model_name="PairwiseProbitModel", num_stim=2) 
        
        setup_message = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str}
        }

        self.s.handle_request(setup_message)

        config_str += self.pre_seed_config(
            "seed_conds", 
            par_ids = ["16", "4", "3"]
        )

        test_fetcher = DataFetcher.from_config(Config(config_str= config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)
        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))

    def test_ex_metadata_match(self):
        validation_set = {7, 15, 47, 50, 53, 57}

        config_str = self.default_config() + self.pre_seed_config("seed_conds", ex_data = {"modality": "vision"})
        test_fetcher = DataFetcher.from_config(Config(config_str = config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)

        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))
        validation_set |= {9, 18, 33, 44, 62}

        config_str = self.default_config() + self.pre_seed_config("seed_conds", ex_data = {"modality": ["vision", "haptics"]})
        test_fetcher = DataFetcher.from_config(Config(config_str = config_str), "my_strat")
        valid_ids = test_fetcher._get_valid_data_ids(self.s)

        self.assertTrue(len(validation_set.intersection(set(valid_ids))) == len(validation_set))

    def test_warm_start(self):
        config_str = self.default_config() + self.pre_seed_config(
            "seed_conds", exp_names=["multi_param_gen", "negative_param_gen"]
        )

        setup_message = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str}
        }

        resume_message = None if self.s.strat_id == -1 else {
            "type": "resume",
            "version": "0.01",
            "message": {"strat_id": self.s.strat_id}
        }

        self.s.handle_request(setup_message)

        for strat in self.s.strat.strat_list:
            if self.s.config.has_option(strat.name, "seed_data_conditions"):
                self.assertFalse(strat._model_is_fresh)
                self.assertTrue(strat.n == 0)
                self.assertTrue(strat._count == 0)

        if resume_message is not None:
            self.s.handle_request(resume_message)

    def test_undefined_parameter_filter(self):
        config_str = self.default_config(par_data={
                    "par1": ["continuous", 0, 1, True],
                    "par2": ["continuous", 0, 1, True]
                },
                outcome_types=["ordinal"], model_name="OrdinalGPModel")\
              + self.pre_seed_config("seed_conds", exp_names=["multi_param_gen", "negative_param_exp"])

        setup_message = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str}
        }        

        resume_message = None if self.s.strat_id == -1 else {
            "type": "resume",
            "version": "0.01",
            "message": {"strat_id": self.s.strat_id}
        }

        self.s.handle_request(setup_message)
        
        for strat in self.s.strat.strat_list:
            if self.s.config.has_option(strat.name, "seed_data_conditions"):
                self.assertFalse(strat._model_is_fresh)
                self.assertFalse(strat.x.le(-1).any())
                self.assertTrue(strat.n == 0)
                self.assertTrue(strat._count == 0)

        if resume_message is not None:
            self.s.handle_request(resume_message)  

    ######################################################
    # this test was initially implemented using a custom profiler module. 
    # As this module was removed so were all references to its code. 
    # If you're interested in reviving this test you'll need to implement profiling 
    # using one of the languages build in profilers. 
    ######################################################
    # def test_filter_2000(self):
    #     config_str = self.default_config() + self.pre_seed_config("seed_conds", 
    #                 exp_ids=["f18d43aa-e4e0-4376-859b-8ef8e9d749dd", "b4e77159-cfd2-45f8-9215-d22c355039e2"])
        
    #     setup_message = {
    #         "type": "setup",
    #         "version": "0.01",
    #         "message": {"config_str": config_str}
    #     }

    #     resume_message = None if self.s.strat_id == -1 else {
    #         "type": "resume",
    #         "version": "0.01",
    #         "message": {"strat_id": self.s.strat_id}
    #     }

    #     self.s.handle_request(setup_message)

    #     if resume_message is not None:
    #         self.s.handle_request(resume_message)
                
