#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.database.tables import DBMasterTable
from aepsych.strategy import Strategy
from aepsych.utils import generate_default_outcome_names
from aepsych.utils_logging import getLogger

logger = getLogger()

# !!!!!! WARNING !!!!!
#
# If you modify the parameter order in the query in _get_data ensure you update these to match!
# The order in which you place the parameters in a sql query dictates the placement in the results
# array.
#
# For example:
# select iteration_id, param_name, param_val, outcome_id results in this list:
#   [iteration_id, param_name, param_val, outcome_id]
#
# !!!!!! WARNING !!!!!
# "const" variable identifiers to avoid magic number indexing into the query results

ITERATION_ID = 0
PARAM_NAME_ID = 1
PARAM_VAL_ID = 2
OUTCOME_ID = 3


class DataFetcher(ConfigurableMixin):
    def __init__(
        self,
        exp_names: Optional[List[str]] = None,
        exp_desc: Optional[List[str]] = None,
        exp_ids: Optional[List[str]] = None,
        par_ids: Optional[List[str]] = None,
        ex_data: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize the DataFetcher object.

        Args:
            exp_names (List[str], optional): A list of experiment names to use as a filter on warm start data.
            exp_desc (List[str], optional): A list of experiment descriptions to use as a filter on warm start data.
            exp_ids (List[str], optional): A list of experiment ids to use as a filter on warm start data.
            par_ids (List[str], optional): A list of participant ids to use as a filter on warm start data.
            ex_data (Dict[str, str], optional): A map of key-value pairs to use as a filter on warm start data.
        """
        self.experiment_names: Optional[List[str]] = exp_names
        self.experiment_desc: Optional[List[str]] = exp_desc
        self.experiment_ids: Optional[List[str]] = exp_ids
        self.participant_ids: Optional[List[str]] = par_ids
        self.extra_metadata: Optional[Dict[str, str]] = ex_data

    @property
    def _has_search_criteria(self) -> bool:
        """Checks to see if the search criteria necessary for warm starting a strategy exists.

        Returns:
            bool: are any filter critera set.
        """
        return (
            self.experiment_ids is not None
            or self.experiment_names is not None
            or self.experiment_desc is not None
            or self.participant_ids is not None
            or self.extra_metadata is not None
        )

    def _build_query(self) -> str:
        query_clauses: List[str] = []
        if self.experiment_names is not None:
            query_clauses.append(
                f"experiment_name in {_print_in_params(self.experiment_names)}"
            )
        if self.experiment_ids is not None:
            query_clauses.append(
                f"experiment_id in {_print_in_params(self.experiment_ids)}"
            )
        if self.experiment_desc is not None:
            query_clauses.append(
                f"experiment_description in {_print_in_params(self.experiment_desc)}"
            )
        if self.participant_ids is not None:
            query_clauses.append(
                f"participant_id in {_print_in_params(self.participant_ids)}"
            )
        if self.extra_metadata is not None:
            query_clauses.append(
                f"{_print_like_params('extra_metadata', self.extra_metadata)}"
            )
        query = "select unique_id from master where "
        for i, clause in enumerate(query_clauses):
            query += clause
            if i < len(query_clauses) - 1:
                query += " and "
        return query

    def _are_stimuli_valid(self, server, config: Config, id: int) -> bool:
        pickeld_stimuli_per_trial = config.get(
            "common", "stimuli_per_trial", fallback=-1
        )
        active_stimuli_per_trial = server.config.get(
            "common", "stimuli_per_trial", fallback=0
        )

        if pickeld_stimuli_per_trial != active_stimuli_per_trial:
            _print_warning(
                self,
                server.db.get_master_record(id),
                f"Data with master id: {id} was matched with {{0}} but is invalid because it has "
                + f"{pickeld_stimuli_per_trial} stimuli and the current experiment has {active_stimuli_per_trial} stimuli. "
                + "Skipping this entry.",
            )

            return False  # trial stimuli don't match this isn't valid data
        return True

    def _are_parameters_valid(self, server, config: Config, id: int) -> bool:
        param_names = set(config.getlist("common", "parnames", element_type=str))
        if len(param_names.intersection(set(server.parnames))) == len(server.parnames):
            match = True
            for param in param_names:
                match = config[param]["par_type"] == server.config[param]["par_type"]
                if not match:
                    break
                if config[param]["par_type"] == "continuous":
                    pickled_lb = config[param]["lower_bound"]
                    pickled_ub = config[param]["upper_bound"]

                    active_lb = server.config[param]["lower_bound"]
                    active_ub = server.config[param]["upper_bound"]
                    if pickled_lb != active_lb or pickled_ub != active_ub:
                        _print_warning(
                            self,
                            server.db.get_master_record(id),
                            f"Data with master id: {id} was matched with {{0}} but its parameter {param} "
                            + f"has bounds ({pickled_lb}, {pickled_ub}) compared to the active bounds "
                            + f"({active_lb}, {active_ub}).",
                        )
            if not match:
                _print_warning(
                    self,
                    server.db.get_master_record(id),
                    f"Data with master id: {id} was matched with {{0}} but is invalid due to parameter "
                    + f"{param} having the type {config[param]['par_type']} while its active type "
                    + f"is {server.config[param]['par_type']}. Skipping this entry.",
                )
                return False  # param types don't match this isn't valid data
        else:
            _print_warning(
                self,
                server.db.get_master_record(id),
                f"Data with master id: {id} was matched with {{0}} but is invalid due to "
                + "a mismatch of parameter names. Skipping this entry.",
            )
            return False  # param names don't match this isn't valid data
        return True

    def _are_outcomes_valid(self, server, config: Config, id: int) -> bool:
        pickled_outcome_types = config.getlist(
            "common", "outcome_types", element_type=str
        )
        active_outcome_types = server.config.getlist(
            "common", "outcome_types", element_type=str
        )

        # this doesn't ensure order i.e. the types may match but could be applied to different outcomes

        outcome_types_match = len(pickled_outcome_types) == len(active_outcome_types)
        for i, outcome_type in enumerate(pickled_outcome_types):
            if not outcome_types_match:
                break
            if outcome_type != active_outcome_types[i]:
                outcome_types_match = False
        if outcome_types_match:
            pickled_outcome_names = set(
                config.getlist(
                    "common",
                    "outcome_names",
                    element_type=str,
                    fallback=generate_default_outcome_names(len(pickled_outcome_types)),
                )
            )

            active_outcome_names = set(
                server.config.getlist(
                    "common",
                    "outcome_names",
                    element_type=str,
                    fallback=generate_default_outcome_names(len(active_outcome_types)),
                )
            )

            if len(pickled_outcome_names.intersection(active_outcome_names)) != len(
                active_outcome_names
            ):
                _print_warning(
                    self,
                    server.db.get_master_record(id),
                    f"Data with master id: {id} was matched with {{0}} but is invalid due to "
                    + "a mismatch in outcome names. Skipping this entry.",
                )
                return False  # outcome types donot match this isn't valid data
        else:
            _print_warning(
                self,
                server.db.get_master_record(id),
                f"Data with master id: {id} was matched with {{0}} but is invalid due to "
                + "a mismatch in outcome types. Skipping this entry.",
            )
            return False  # outcome names dont match this isn't valid data
        return True

    def _is_ex_metadata_match_valid(self, config: Config) -> bool:
        if not self.extra_metadata:
            return True
        pickled_ex_data = config.get_metadata(only_extra=True)
        if len(pickled_ex_data) == 0:
            return True
        potential_matches = pickled_ex_data.keys() & self.extra_metadata.keys()
        for key in potential_matches:
            if pickled_ex_data[key] == self.extra_metadata[key] or (
                isinstance(self.extra_metadata[key], Iterable)
                and pickled_ex_data[key] in self.extra_metadata[key]
            ):
                return True
        # don't warn about skipping this data since it was part of the initial filter criteria and
        # we don't warn about any of that filtering.

        return False

    def _get_valid_data_ids(self, server) -> List[int]:
        """Gets all master table ids associated with the data defined by provided search criteria.
           The data is then filtered by the current strategy's properties to see if it would be valid for use.

        Args:
            server (AEPsychServer): the instance of the server.

        Returns:
            List[int]: a list of master table ids that meet all criteria to be valid for use in the current strategy.
        """
        valid_match_ids = []
        query = self._build_query()
        potential_match_ids = server.db.execute_sql_query(query, None)

        for id in potential_match_ids:
            config: Config = server.db.get_config_for(id[0])

            # ensure that any data matched on metadata keys actually have matching values in their configs.

            if not self._is_ex_metadata_match_valid(config):
                continue
            # check that stimuli_per_trial match between current + pickled config

            if not self._are_stimuli_valid(server, config, id[0]):
                continue
            # check param names + types match between current + pickeld config

            if not self._are_parameters_valid(server, config, id[0]):
                continue
            # check outcome names + types match between current + pickled config

            if not self._are_outcomes_valid(server, config, id[0]):
                continue
            # we found a valid match keep a record of it

            valid_match_ids.append(id[0])
        return valid_match_ids

    def _construct_data_query(self, ids: List[int]) -> str:
        # !!!!!! WARNING !!!!!
        # If you modify the parameter order in the query
        # ensure you update the "const" id values to match!
        #
        # see comment at line 45
        # !!!!!! WARNING !!!!!

        return f"""select param_data.iteration_id, param_data.param_name,
                param_data.param_value, outcome_data.outcome_value from param_data
                inner join outcome_data on param_data.iteration_id = outcome_data.iteration_id
                inner join raw_data on param_data.iteration_id = raw_data.unique_id
                inner join master on raw_data.master_table_id = master.unique_id
                where master.unique_id in {_print_in_params(ids)}"""

    def _get_data(self, server, ids: List[int]) -> List[Tuple[Any]]:
        """Gets the actual data to be fed into the strategy's tensors.

        Args:
            server (AEPsychServer): The instance of the server.
            ids (List[int]): A list of master table ids used to identify and retrive experiment data.

        Returns:
            results (List[Tuple[Any]]): a list of all data to be fed into the strategy's tensors.
        """
        query = self._construct_data_query(ids)
        results = server.db.execute_sql_query(query, None)
        return results if results is not None else []

    def warm_start_strat(self, server, strat: Strategy):
        """Warm start the current strategy with data from previous experiments.

        Args:
            server (AEPsychServer): The instance of the server
            strat (Strategy): The strategy to warm start.
        """
        if not self._has_search_criteria:
            return  # no data to process
        valid_match_ids = self._get_valid_data_ids(server)
        data = self._get_data(server, valid_match_ids)

        # since we append param names with _stimuli + n when writing the record to the database
        # we need to parse the param name to ensure it will match the names stored in server.parname
        # otherwise server._config_to_tensor will fail.

        def trim_param_name(name: str) -> str:
            end = name.rfind("_stimuli")
            if end == -1:
                end = None
            return name[0:end]

        i = 0
        fetch_count = 0
        while i < len(data):
            # recreate the config dictionary ensuring that all related stimuli are grouped
            # together. This way we can ensure they're being fed into the model correctly.

            config = {
                trim_param_name(data[i][PARAM_NAME_ID]): float(data[i][PARAM_VAL_ID])
            }
            outcome = torch.tensor(data[i][OUTCOME_ID], dtype=torch.float64)

            # while the parameters iteration id matches keep adding data to the dictionary
            # we can do this since all data is pre-sorted by iteration when fed into the database
            # and maintains that sorting when pulled from it.

            j = 1
            while (
                i + j < len(data) and data[i][ITERATION_ID] == data[i + j][ITERATION_ID]
            ):
                config[trim_param_name(data[i + j][PARAM_NAME_ID])] = float(
                    data[i + j][PARAM_VAL_ID]
                )
                j += 1

            i += j
            x = server._config_to_tensor(config)

            # only use data to warm model if it is still defined after passing it through
            # the strategy's transforms.

            res = strat.transforms.transform(x)
            if not (res.isinf().any() or res.isnan().any()):
                fetch_count += 1
                strat.pre_warm_model(x, outcome)
        if len(data) - fetch_count > 0:
            logger.warning(
                f"""{len(data) - fetch_count} rows of data had parameters that were undefined in the bounds 
                of the current experiment, discarding."""
            )
        logger.info(
            f"Strategy {strat.name} was warm started with {fetch_count} rows of data."
        )

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Populate an instance of DataFetcher with search criteria provided in the experiment's config.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): The name of the strategy to warm start (Not actually optional here.)
            options (Dict[str, Any], optional): options are ignored.

        Raises:
            ValueError: the name of the strategy is necessary to identify warm start search criteria.
            KeyError: the config specified this strategy should be warm started but the associated config section wasn't defined.

        Returns:
            Dict[str, Any]: a dictionary of the search criteria described in the experiment's config
        """
        if name is None:
            raise ValueError("name of strategy must be set to warm start strategy.")
        if not config.has_option(name, "seed_data_conditions"):
            return {
                "exp_names": None,
                "exp_ids": None,
                "exp_desc": None,
                "par_ids": None,
                "ex_data": None,
            }
        experiment_names = None
        experiment_ids = None
        experiment_descriptions = None
        participant_ids = None
        extra_metadata = None

        seed_conds_name = config.get(name, "seed_data_conditions")
        if config.has_section(seed_conds_name):
            seed_conds_data = config.to_dict()[seed_conds_name].copy()
            if "experiment_name" in seed_conds_data:
                experiment_names = config.getlist(
                    seed_conds_name, "experiment_name", element_type=str
                )
                seed_conds_data.pop("experiment_name", None)
            if "experiment_id" in seed_conds_data:
                experiment_ids = config.getlist(
                    seed_conds_name, "experiment_id", element_type=str
                )
                seed_conds_data.pop("experiment_id", None)
            if "experiment_description" in seed_conds_data:
                experiment_descriptions = config.getlist(
                    seed_conds_name, "experiment_description", element_type=str
                )
                seed_conds_data.pop("experiment_description", None)
            if "participant_id" in seed_conds_data:
                participant_ids = config.getlist(
                    seed_conds_name, "participant_id", element_type=str
                )
                seed_conds_data.pop("participant_id", None)
            extra_metadata = seed_conds_data if len(seed_conds_data) > 0 else None
        else:
            raise KeyError(
                f"config must have section {seed_conds_name} to warm start strategy."
            )
        return {
            "exp_names": experiment_names,
            "exp_ids": experiment_ids,
            "exp_desc": experiment_descriptions,
            "par_ids": participant_ids,
            "ex_data": extra_metadata,
        }


def _print_in_params(list_to_print: List[Any]) -> str:
    # This prints a list of values to be used in a sql in statement

    out_string = "("
    for i, val in enumerate(list_to_print):
        out_string += f"'{val}'"
        if i < len(list_to_print) - 1:
            out_string += ", "
    out_string += ")"
    return out_string


def _print_like_params(column_name: str, dict_to_print: Dict[str, str]) -> str:
    # This prints a list of values to be in a sequence of sql like statments
    # that are combined using a binary or operation.

    out_string = "("
    for i, (key, val) in enumerate(dict_to_print.items()):
        out_string += f"{column_name} like '%\"{key}\":%'"
        if i < len(dict_to_print) - 1:
            out_string += " or "
    out_string += ")"
    return out_string


def _print_warning(fetcher: DataFetcher, query_record: DBMasterTable, format: str):
    first_match = True
    match_str = ""
    if (
        fetcher.experiment_names
        and query_record.experiment_name in fetcher.experiment_names
    ):
        match_str += f"experiment_name = {query_record.experiment_name}"
        first_match = False
    if (
        fetcher.experiment_desc
        and query_record.experiment_description in fetcher.experiment_desc
    ):
        if not first_match:
            match_str += " and "
        match_str += f"experimend_desc = {query_record.experiment_description}"
        first_match = False
    if fetcher.experiment_ids and query_record.experiment_id in fetcher.experiment_ids:
        if not first_match:
            match_str += " and "
        match_str += f"experimend_id = {query_record.experiment_id}"
        first_match = False
    if (
        fetcher.participant_ids
        and query_record.participant_id in fetcher.participant_ids
    ):
        if not first_match:
            match_str += " and "
        match_str += f"participant_id = {query_record.participant_id}"
        first_match = False
    # assumes that all false positive metadata matches have been filtered out.

    if fetcher.extra_metadata and query_record.extra_metadata:
        ex_data_str = "" if first_match else " and "
        ex_data = json.loads(query_record.extra_metadata)

        first_ex_match = True
        for i, (key, val) in enumerate(fetcher.extra_metadata.items()):
            if key in ex_data and (
                val == ex_data[key]
                or (isinstance(val, Iterable) and ex_data[key] in val)
            ):
                if not first_ex_match:
                    ex_data_str = ex_data_str + ", "
                    if i == len(fetcher.extra_metadata) - 1:
                        ex_data_str += " and "
                ex_data_str += f"{key} = {val}"
                first_ex_match = False
        if not first_ex_match:
            match_str += ex_data_str
    logger.warning(format.format(match_str))
