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
from typing import Dict

import aepsych.database.tables as tables
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


logger = logging.getLogger()


class Database:
    def __init__(self, db_path=None):
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

    def get_engine(self):
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

    def delete_db(self):
        if self._engine is not None and self._full_db_path.exists():
            os.remove(self._full_db_path.as_posix())
            self._engine = None

    def is_update_required(self):
        return (
            tables.DBMasterTable.requires_update(self._engine)
            or tables.DbReplayTable.requires_update(self._engine)
            or tables.DbStratTable.requires_update(self._engine)
            or tables.DbConfigTable.requires_update(self._engine)
        )

    def perform_updates(self):
        """Perform updates on known tables. SQLAlchemy doesn't do alters so they're done the old fashioned way."""
        tables.DBMasterTable.update(self._engine)
        tables.DbReplayTable.update(self._engine)
        tables.DbStratTable.update(self._engine)
        tables.DbConfigTable.update(self._engine)

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
    def execute_sql_query(self, query: str, vals: Dict[str, str]):
        """Execute an arbitrary query written in sql."""
        with self.session_scope() as session:
            return session.execute(query, vals).fetchall()

    def get_master_records(self):
        """Grab the list of master records."""
        records = self._session.query(tables.DBMasterTable).all()
        return records

    def get_master_record(self, experiment_id):
        """Grab the list of master record for a specific experiment (master) id."""
        records = (
            self._session.query(tables.DBMasterTable)
            .filter(tables.DBMasterTable.experiment_id == experiment_id)
            .all()
        )

        if 0 < len(records):
            return records[0]

        return None

    def get_replay_for(self, master_id):
        """Get the replay records for a specific master row."""
        master_record = self.get_master_record(master_id)

        if master_record is not None:
            return master_record.children_replay

        return None

    def get_strats_for(self, master_id=0):
        """Get the strat records for a specific master row."""
        master_record = self.get_master_record(master_id)

        if master_record is not None and len(master_record.children_strat) > 0:
            return [c.strat for c in master_record.children_strat]

        return None

    def get_strat_for(self, master_id, strat_id=-1):
        """Get a specific strat record for a specific master row."""
        master_record = self.get_master_record(master_id)

        if master_record is not None and len(master_record.children_strat) > 0:
            return master_record.children_strat[strat_id].strat

        return None

    def get_config_for(self, master_id):
        """Get the strat records for a specific master row."""
        master_record = self.get_master_record(master_id)

        if master_record is not None:
            return master_record.children_config[0].config
        return None

    def record_setup(self, description, name, id=None, request=None) -> str:
        self.get_engine()

        if id is None:
            master_table = tables.DBMasterTable()
            master_table.experiment_description = description
            master_table.experiment_name = name
            master_table.experiment_id = str(uuid.uuid4())

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

    def record_message(self, master_table, type, request) -> None:
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

    def record_strat(self, master_table, strat):
        strat_entry = tables.DbStratTable()
        strat_entry.strat = strat
        strat_entry.timestamp = datetime.datetime.now()
        strat_entry.parent = master_table

        self._session.add(strat_entry)
        self._session.commit()

    def record_config(self, master_table, config):
        config_entry = tables.DbConfigTable()
        config_entry.config = config
        config_entry.timestamp = datetime.datetime.now()
        config_entry.parent = master_table

        self._session.add(config_entry)
        self._session.commit()

    def list_master_records(self):
        master_records = self.get_master_records()

        print("Listing master records:")
        for record in master_records:
            print(
                f'\t{record.unique_id} - name: "{record.experiment_name}" experiment id: {record.experiment_id}'
            )
