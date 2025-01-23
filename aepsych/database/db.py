#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import aepsych.database.tables as tables
import pandas as pd
from aepsych.config import Config
from aepsych.strategy import Strategy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import close_all_sessions

logger = logging.getLogger()


class Database:
    def __init__(self, db_path: Optional[str] = None, update: bool = True) -> None:
        """Initialize the database object.

        Args:
            db_path (str, optional): The path to the database. Defaults to None.
            update (bool): Update the db to the latest schema. Defaults to True.
        """
        if db_path is None:
            db_path = "./databases/default.db"

        db_dir, db_name = os.path.split(db_path)
        self._db_name = db_name
        self._db_dir = db_dir

        if os.path.exists(db_path):
            logger.info(f"Found DB at {db_path}, appending!")
        else:
            logger.info(f"No DB found at {db_path}, creating a new DB!")

        self._engine = self.get_engine()

        if update and self.is_update_required():
            self.perform_updates()

    def get_engine(self) -> sessionmaker:
        """Get the engine for the database.

        Returns:
            sessionmaker: The sessionmaker object for the database.
        """
        if not hasattr(self, "_engine") or self._engine is None:
            self._full_db_path = Path(self._db_dir)
            self._full_db_path.mkdir(parents=True, exist_ok=True)
            self._full_db_path = self._full_db_path.joinpath(self._db_name)

            self._engine = create_engine(f"sqlite:///{self._full_db_path.as_posix()}")

            # create the table metadata and tables
            tables.Base.metadata.create_all(self._engine)

            # create an ongoing session to be used. Provides a conduit
            # to the db so the instantiated objects work properly.
            Session = sessionmaker(bind=self.get_engine())
            self._session = Session()

        return self._engine

    def delete_db(self) -> None:
        """Delete the database."""
        if self._engine is not None and self._full_db_path.exists():
            close_all_sessions()
            self._full_db_path.unlink()
            self._engine = None

    def is_update_required(self) -> bool:
        """Check if an update is required on the database.

        Returns:
            bool: True if an update is required, False otherwise.
        """
        return (
            tables.DBMasterTable.requires_update(self._engine)
            or tables.DbReplayTable.requires_update(self._engine)
            or tables.DbStratTable.requires_update(self._engine)
            or tables.DbConfigTable.requires_update(self._engine)
            or tables.DbRawTable.requires_update(self._engine)
            or tables.DbParamTable.requires_update(self._engine)
            or tables.DbOutcomeTable.requires_update(self._engine)
        )

    def perform_updates(self) -> None:
        """Perform updates on known tables. SQLAlchemy doesn't do alters so they're done the old fashioned way."""
        tables.DBMasterTable.update(self._engine)
        tables.DbReplayTable.update(self._engine)
        tables.DbStratTable.update(self._engine)
        tables.DbConfigTable.update(self._engine)
        tables.DbRawTable.update(self, self._engine)
        tables.DbParamTable.update(self._engine)
        tables.DbOutcomeTable.update(self._engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        Session = sessionmaker(bind=self.get_engine())
        session = Session()
        try:
            yield session
            session.commit()
        except Exception as err:
            logger.error(f"db session use failed: {err}")
            session.rollback()
            raise
        finally:
            session.close()

    # @retry(stop_max_attempt_number=8, wait_exponential_multiplier=1.8)
    def execute_sql_query(self, query: str, vals: Dict[str, str]) -> List[Any]:
        """Execute an arbitrary query written in sql.

        Args:
            query (str): The query to execute.
            vals (Dict[str, str]): The values to use in the query.

        Returns:
            List[Any]: The results of the query.
        """
        with self.session_scope() as session:
            return session.execute(query, vals).all()

    def get_master_records(self) -> List[tables.DBMasterTable]:
        """Grab the list of master records.

        Returns:
            List[tables.DBMasterTable]: The list of master records.
        """
        records = self._session.query(tables.DBMasterTable).all()
        return records

    def get_master_record(self, master_id: int) -> Optional[tables.DBMasterTable]:
        """Grab the list of master record for a specific master id (uniquie_id of master table).

        Args:
            master_id (int): The master_id, which is the master key of the master table.

        Returns:
            tables.DBMasterTable or None: The master record or None if it doesn't exist.
        """
        records = (
            self._session.query(tables.DBMasterTable)
            .filter(tables.DBMasterTable.unique_id == master_id)
            .all()
        )

        if 0 < len(records):
            return records[0]

        return None

    def get_replay_for(self, master_id: int) -> Optional[List[tables.DbReplayTable]]:
        """Get the replay records for a specific master row.

        Args:
            master_id (int): The unique id for the master row (it's the master key).

        Returns:
            List[tables.DbReplayTable] or None: The replay records or None if they don't exist.
        """
        master_record = self.get_master_record(master_id)

        if master_record is not None:
            return master_record.children_replay

        return None

    def get_strats_for(self, master_id: int = 0) -> Optional[List[Any]]:
        """Get the strat records for a specific master row.

        Args:
            master_id (int): The master table unique ID. Defaults to 0.

        Returns:
            List[Any] or None: The strat records or None if they don't exist.
        """
        master_record = self.get_master_record(master_id)

        if master_record is not None and len(master_record.children_strat) > 0:
            return [c.strat for c in master_record.children_strat]

        return None

    def get_strat_for(self, master_id: int, strat_id: int = -1) -> Optional[Any]:
        """Get a specific strat record for a specific master row.

        Args:
            master_id (int): The master id.
            strat_id (int): The strat id. Defaults to -1.

        Returns:
            Any: The strat record.
        """
        master_record = self.get_master_record(master_id)

        if master_record is not None and len(master_record.children_strat) > 0:
            return master_record.children_strat[strat_id].strat

        return None

    def get_config_for(self, master_id: int) -> Optional[Any]:
        """Get the strat records for a specific master row.

        Args:
            master_id (int): The master id.

        Returns:
            Any: The config records.
        """
        master_record = self.get_master_record(master_id)

        if master_record is not None:
            return master_record.children_config[0].config
        return None

    def get_raw_for(self, master_id: int) -> Optional[List[tables.DbRawTable]]:
        """Get the raw data for a specific master row.

        Args:
            master_id (int): The master id.

        Returns:
            List[tables.DbRawTable] or None: The raw data or None if it doesn't exist.
        """
        master_record = self.get_master_record(master_id)

        if master_record is not None:
            return master_record.children_raw

        return None

    def get_params_for(self, master_id: int) -> List[List[tables.DbParamTable]]:
        """Get the rows of the parameter table for the master_id's experiment. Each
        row contains data for a certain trial: the parameter name and its values.
        If a trial has multiple parameters, there will be multiple rows for that trial.
        Trials are delineated by the iteration_id.

        Args:
            master_id (int): The master id.

        Returns:
            List[List[tables.DbParamTable]]: The parameters as a list of lists, where each inner list represents one trial.
        """
        raw_record = self.get_raw_for(master_id)

        if raw_record is not None:
            return [
                rec.children_param
                for rec in self.get_raw_for(master_id)
                if rec is not None
            ]

        return []

    def get_outcomes_for(self, master_id: int) -> List[List[tables.DbParamTable]]:
        """Get the rows of the outcome table for the master_id's experiment. Each
        row contains data for a certain trial: the outcome name and its values.
        If a trial has multiple outcomes, there will be multiple rows for that trial.
        Trials are delineated by the iteration_id.

        Args:
            master_id (int): The master id.

        Returns:
            List[List[tables.DbOutcomeTable]]: The outcomes as a list of lists, where each inner list represents one trial.
        """
        raw_record = self.get_raw_for(master_id)

        if raw_record is not None:
            return [
                rec.children_outcome
                for rec in self.get_raw_for(master_id)
                if rec is not None
            ]

        return []

    def record_setup(
        self,
        description: str = None,
        name: str = None,
        extra_metadata: Optional[str] = None,
        exp_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        par_id: Optional[int] = None,
    ) -> str:
        """Record the setup of an experiment.

        Args:
            description (str, optional): The description of the experiment, defaults to None.
            name (str, optional): The name of the experiment, defaults to None.
            extra_metadata (str, optional): Extra metadata. Defaults to None.
            exp_id (str, optional): The id of the experiment. Defaults to a generated uuid.
            request (Dict[str, Any], optional): The request. Defaults to None.
            par_id (int, optional): The participant id. Defaults to generated uuid.

        Returns:
            str: The experiment id.
        """
        self.get_engine()

        master_table = tables.DBMasterTable()
        master_table.experiment_description = description
        master_table.experiment_name = name
        master_table.experiment_id = exp_id if exp_id is not None else str(uuid.uuid4())
        master_table.participant_id = (
            par_id if par_id is not None else str(uuid.uuid4())
        )
        master_table.extra_metadata = extra_metadata
        self._session.add(master_table)

        logger.debug(f"record_setup = [{master_table}]")

        record = tables.DbReplayTable()
        record.message_type = "setup"
        record.message_contents = request

        if request is not None and "extra_info" in request:
            record.extra_info = request["extra_info"]

        record.timestamp = datetime.datetime.now()
        record.parent = master_table
        logger.debug(f"record_setup = [{record}]")

        self._session.add(record)
        self._session.commit()

        # return the master table if it has a link to the list of child rows
        # tis needs to be passed into all future calls to link properly
        return master_table

    def record_message(
        self, master_table: tables.DBMasterTable, type: str, request: Dict[str, Any]
    ) -> None:
        """Record a message in the database.

        Args:
            master_table (tables.DBMasterTable): The master table.
            type (str): The type of the message.
            request (Dict[str, Any]): The request.
        """
        # create a linked setup table
        record = tables.DbReplayTable()
        record.message_type = type
        record.message_contents = request

        if "extra_info" in request:
            record.extra_info = request["extra_info"]

        record.timestamp = datetime.datetime.now()
        record.parent = master_table

        self._session.add(record)
        self._session.commit()

    def record_raw(
        self,
        master_table: tables.DBMasterTable,
        model_data: Any,
        timestamp: Optional[datetime.datetime] = None,
        **extra_data,
    ) -> tables.DbRawTable:
        """Record raw data in the database.

        Args:
            master_table (tables.DBMasterTable): The master table.
            model_data (Any): The model data.
            timestamp (datetime.datetime, optional): The timestamp. Defaults to None.
            **extra_data: Extra data to save as a json in the raw.

        Returns:
            tables.DbRawTable: The raw entry.
        """
        raw_entry = tables.DbRawTable()
        raw_entry.model_data = model_data

        if timestamp is None:
            raw_entry.timestamp = datetime.datetime.now()
        else:
            raw_entry.timestamp = timestamp
        raw_entry.parent = master_table

        raw_entry.extra_data = json.dumps(extra_data)

        self._session.add(raw_entry)
        self._session.commit()

        return raw_entry

    def record_param(
        self, raw_table: tables.DbRawTable, param_name: str, param_value: str
    ) -> None:
        """Record a parameter in the database.

        Args:
            raw_table (tables.DbRawTable): The raw table.
            param_name (str): The parameter name.
            param_value (str): The parameter value.
        """
        param_entry = tables.DbParamTable()
        param_entry.param_name = param_name
        param_entry.param_value = param_value

        param_entry.parent = raw_table

        self._session.add(param_entry)
        self._session.commit()

    def record_outcome(
        self, raw_table: tables.DbRawTable, outcome_name: str, outcome_value: float
    ) -> None:
        """Record an outcome in the database.

        Args:
            raw_table (tables.DbRawTable): The raw table.
            outcome_name (str): The outcome name.
            outcome_value (float): The outcome value.
        """
        outcome_entry = tables.DbOutcomeTable()
        outcome_entry.outcome_name = outcome_name
        outcome_entry.outcome_value = outcome_value

        outcome_entry.parent = raw_table

        self._session.add(outcome_entry)
        self._session.commit()

    def record_strat(self, master_table: tables.DBMasterTable, strat: Strategy) -> None:
        """Record a strategy in the database.

        Args:
            master_table (tables.DBMasterTable): The master table.
            strat (Strategy): The strategy.
        """
        strat_entry = tables.DbStratTable()
        strat_entry.strat = strat
        strat_entry.timestamp = datetime.datetime.now()
        strat_entry.parent = master_table

        self._session.add(strat_entry)
        self._session.commit()

    def record_config(self, master_table: tables.DBMasterTable, config: Config) -> None:
        """Record a config in the database.

        Args:
            master_table (tables.DBMasterTable): The master table.
            config (Config): The config.
        """
        config_entry = tables.DbConfigTable()
        config_entry.config = config
        config_entry.timestamp = datetime.datetime.now()
        config_entry.parent = master_table

        self._session.add(config_entry)
        self._session.commit()

    def summarize_experiments(self) -> pd.DataFrame:
        """Provides a summary of the experiments contained in the database as a pandas dataframe.

        This function can also be called from the command line using
            `aepsych_database --db PATH_TO_DB --summarize`

        Returns:
            pandas.Dataframe: The dataframe containing the summary info.
        """

        def get_parnames(master_id):
            config = self.get_config_for(master_id)
            return set(config.getlist("common", "parnames", element_type=str))

        def get_outcome_names(master_id):
            config = self.get_config_for(master_id)
            outcome_types = config.getlist("common", "outcome_types", element_type=str)

            def get_fallback_names(count: int):
                if count == 1:
                    return ["outcome"]

                return ["outcome_" + i for i in range(count)]

            return set(
                config.getlist(
                    "common",
                    "outcome_names",
                    element_type=str,
                    fallback=get_fallback_names(len(outcome_types)),
                )
            )

        def get_stimuli_per_trial(master_id):
            config = self.get_config_for(master_id)
            return config.getint("common", "stimuli_per_trial")

        records = self.get_master_records()
        exp_dict = {
            "experiment_id": [rec.experiment_id for rec in records],
            "experiment_name": [rec.experiment_name for rec in records],
            "experiment_description": [rec.experiment_description for rec in records],
            "participant_id": [rec.participant_id for rec in records],
        }

        extra_metadata = [
            json.loads(rec.extra_metadata) if rec.extra_metadata is not None else {}
            for rec in records
        ]
        keys = {key for met in extra_metadata for key in met}

        for key in keys:
            exp_dict[key] = [met[key] if key in met else None for met in extra_metadata]

        exp_dict.update(
            {
                "creation_time": [
                    self.get_replay_for(rec.unique_id)[0].timestamp for rec in records
                ],
                "time_last_modified": [
                    self.get_replay_for(rec.unique_id)[-1].timestamp for rec in records
                ],
                "stimuli_per_trial": [
                    get_stimuli_per_trial(rec.unique_id) for rec in records
                ],
                "parameter_names": [get_parnames(rec.unique_id) for rec in records],
                "outcome_names": [get_outcome_names(rec.unique_id) for rec in records],
                "n_data": [
                    len(self.get_outcomes_for(rec.unique_id)) for rec in records
                ],
            }
        )

        return pd.DataFrame(exp_dict)

    def get_data_frame(self) -> pd.DataFrame:
        """Converts parameter and outcome data in the database into a pandas dataframe.

        Returns:
            pandas.Dataframe: The dataframe containing the parameter and outcome data.
        """
        records = self.get_master_records()
        dfs = []
        for rec in records:
            rows = []
            parameters = self.get_params_for(rec.unique_id)
            outcomes = self.get_outcomes_for(rec.unique_id)
            for pars, outs in zip(parameters, outcomes):
                row = {}
                row["experiment_id"] = rec.experiment_id
                row["experiment_name"] = rec.experiment_name
                row["experiment_description"] = rec.experiment_description
                row["participant_id"] = rec.participant_id
                row["timestamp"] = pars[0].parent.timestamp

                row.update({par.param_name: par.param_value for par in pars})
                row.update({out.outcome_name: out.outcome_value for out in outs})

                rows.append(row)

            df = pd.DataFrame(rows)
            dfs.append(df)

        return pd.concat(dfs)

    def to_csv(self, path: str):
        """Exports the parameter and outcome data in the database to a csv file.

        This function can also be called from the command line using
            `aepsych_database --db PATH_TO_DB --tocsv PATH_TO_CSV`

        Args:
            path (str): The filepath of the output csv.
        """
        df = self.get_data_frame()
        df.to_csv(path, index=False)
