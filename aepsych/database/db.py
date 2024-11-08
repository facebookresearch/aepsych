#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import aepsych.database.tables as tables
from aepsych.config import Config
from aepsych.strategy import Strategy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import close_all_sessions

logger = logging.getLogger()


class Database:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the database object.

        Args:
            db_path (str, optional): The path to the database. Defaults to None.
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
            return session.execute(query, vals).fetchall()

    def get_master_records(self) -> List[tables.DBMasterTable]:
        """Grab the list of master records.

        Returns:
            List[tables.DBMasterTable]: The list of master records.
        """
        records = self._session.query(tables.DBMasterTable).all()
        return records

    def get_master_record(self, experiment_id: int) -> Optional[tables.DBMasterTable]:
        """Grab the list of master record for a specific experiment (master) id.

        Args:
            experiment_id (int): The experiment id.

        Returns:
            tables.DBMasterTable or None: The master record or None if it doesn't exist.
        """
        records = (
            self._session.query(tables.DBMasterTable)
            .filter(tables.DBMasterTable.experiment_id == experiment_id)
            .all()
        )

        if 0 < len(records):
            return records[0]

        return None

    def get_replay_for(self, master_id: int) -> Optional[List[tables.DbReplayTable]]:
        """Get the replay records for a specific master row.

        Args:
            master_id (int): The master id.

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
            master_id (int): The master id. Defaults to 0.

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

    def get_all_params_for(self, master_id: int) -> Optional[List[tables.DbRawTable]]:
        """Get the parameters for all the iterations of a specific experiment.

        Args:
            master_id (int): The master id.

        Returns:
            List[tables.DbRawTable] or None: The parameters or None if they don't exist.
        """
        raw_record = self.get_raw_for(master_id)
        params = []

        if raw_record is not None:
            for raw in raw_record:
                for param in raw.children_param:
                    params.append(param)
            return params

        return None

    def get_param_for(
        self, master_id: int, iteration_id: int
    ) -> Optional[List[tables.DbRawTable]]:
        """Get the parameters for a specific iteration of a specific experiment.

        Args:
            master_id (int): The master id.
            iteration_id (int): The iteration id.

        Returns:
            List[tables.DbRawTable] or None: The parameters or None if they don't exist.
        """
        raw_record = self.get_raw_for(master_id)

        if raw_record is not None:
            for raw in raw_record:
                if raw.unique_id == iteration_id:
                    return raw.children_param

        return None

    def get_all_outcomes_for(self, master_id: int) -> Optional[List[tables.DbRawTable]]:
        """Get the outcomes for all the iterations of a specific experiment.

        Args:
            master_id (int): The master id.

        Returns:
            List[tables.DbRawTable] or None: The outcomes or None if they don't exist.
        """
        raw_record = self.get_raw_for(master_id)
        outcomes = []

        if raw_record is not None:
            for raw in raw_record:
                for outcome in raw.children_outcome:
                    outcomes.append(outcome)
            return outcomes

        return None

    def get_outcome_for(
        self, master_id: int, iteration_id: int
    ) -> Optional[List[tables.DbRawTable]]:
        """Get the outcomes for a specific iteration of a specific experiment.

        Args:
            master_id (int): The master id.
            iteration_id (int): The iteration id.

        Returns:
            List[tables.DbRawTable] or None: The outcomes or None if they don't exist.
        """
        raw_record = self.get_raw_for(master_id)

        if raw_record is not None:
            for raw in raw_record:
                if raw.unique_id == iteration_id:
                    return raw.children_outcome

        return None

    def record_setup(
        self,
        description: str,
        name: str,
        extra_metadata: Optional[str] = None,
        id: Optional[str] = None,
        request: Dict[str, Any] = None,
        participant_id: Optional[int] = None,
    ) -> str:
        """Record the setup of an experiment.

        Args:
            description (str): The description of the experiment.
            name (str): The name of the experiment.
            extra_metadata (str, optional): Extra metadata. Defaults to None.
            id (str, optional): The id of the experiment. Defaults to None.
            request (Dict[str, Any]): The request. Defaults to None.
            participant_id (int, optional): The participant id. Defaults to None.

        Returns:
            str: The experiment id.
        """
        self.get_engine()

        if id is None:
            master_table = tables.DBMasterTable()
            master_table.experiment_description = description
            master_table.experiment_name = name
            master_table.experiment_id = str(uuid.uuid4())
            if participant_id is not None:
                master_table.participant_id = participant_id
            else:
                master_table.participant_id = str(
                    uuid.uuid4()
                )  # no p_id specified will result in a generated UUID

            master_table.extra_metadata = extra_metadata

            self._session.add(master_table)

            logger.debug(f"record_setup = [{master_table}]")
        else:
            master_table = self.get_master_record(id)
            if master_table is None:
                raise RuntimeError(f"experiment id {id} doesn't exist in the db.")

        record = tables.DbReplayTable()
        record.message_type = "setup"
        record.message_contents = request

        if "extra_info" in request:
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
    ) -> tables.DbRawTable:
        """Record raw data in the database.

        Args:
            master_table (tables.DBMasterTable): The master table.
            model_data (Any): The model data.
            timestamp (datetime.datetime, optional): The timestamp. Defaults to None.

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

    def list_master_records(self) -> None:
        """List the master records."""
        master_records = self.get_master_records()

        print("Listing master records:")
        for record in master_records:
            print(
                f'\t{record.unique_id} - name: "{record.experiment_name}" experiment id: {record.experiment_id}'
            )
