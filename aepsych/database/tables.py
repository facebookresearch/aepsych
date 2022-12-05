#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pickle
from collections.abc import Iterable

from aepsych.config import Config
from aepsych.version import __version__
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    String,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

logger = logging.getLogger()

Base = declarative_base()

"""
Original Schema
CREATE TABLE master (
unique_id INTEGER NOT NULL,
experiment_name VARCHAR(256),
experiment_description VARCHAR(2048),
experiment_id VARCHAR(10),
PRIMARY KEY (unique_id),
UNIQUE (experiment_id)
);
CREATE TABLE replay_data (
unique_id INTEGER NOT NULL,
timestamp DATETIME,
message_type VARCHAR(64),
message_contents BLOB,
master_table_id INTEGER,
PRIMARY KEY (unique_id),
FOREIGN KEY(master_table_id) REFERENCES master (unique_id)
);
"""


class DBMasterTable(Base):
    """
    Master table to keep track of all experiments and unique keys associated with the experiment
    """

    __tablename__ = "master"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(256))
    experiment_description = Column(String(2048))
    experiment_id = Column(String(10), unique=True)
    participant_id = Column(String(50), unique=True)

    extra_metadata = Column(String(4096))  # JSON-formatted metadata

    children_replay = relationship("DbReplayTable", back_populates="parent")
    children_strat = relationship("DbStratTable", back_populates="parent")
    children_config = relationship("DbConfigTable", back_populates="parent")
    children_raw = relationship("DbRawTable", back_populates="parent")

    @classmethod
    def from_sqlite(cls, row):
        this = DBMasterTable()
        this.unique_id = row["unique_id"]
        this.experiment_name = row["experiment_name"]
        this.experiment_description = row["experiment_description"]
        this.experiment_id = row["experiment_id"]
        return this

    def __repr__(self):
        return (
            f"<DBMasterTable(unique_id={self.unique_id})"
            f", experiment_name={self.experiment_name}, "
            f"experiment_description={self.experiment_description}, "
            f"experiment_id={self.experiment_id})>"
        )

    @staticmethod
    def update(engine):
        logger.info("DBMasterTable : update called")
        if not DBMasterTable._has_column(engine, "extra_metadata"):
            DBMasterTable._add_column(engine, "extra_metadata")
        if not DBMasterTable._has_column(engine, "participant_id"):
            DBMasterTable._add_column(engine, "participant_id")

    @staticmethod
    def requires_update(engine):
        return not DBMasterTable._has_column(
            engine, "extra_metadata"
        ) or not DBMasterTable._has_column(engine, "participant_id")

    @staticmethod
    def _has_column(engine, column: str):
        result = engine.execute(
            "SELECT COUNT(*) FROM pragma_table_info('master') WHERE name='{0}'".format(
                column
            )
        )
        rows = result.fetchall()
        count = rows[0][0]
        return count != 0

    @staticmethod
    def _add_column(engine, column: str):
        try:
            result = engine.execute(
                "SELECT COUNT(*) FROM pragma_table_info('master') WHERE name='{0}'".format(
                    column
                )
            )
            rows = result.fetchall()
            count = rows[0][0]

            if 0 == count:
                logger.debug(
                    "Altering the master table to add the {0} column".format(column)
                )
                engine.execute(
                    "ALTER TABLE master ADD COLUMN {0} VARCHAR".format(column)
                )
                engine.commit()
        except Exception as e:
            logger.debug(f"Column already exists, no need to alter. [{e}]")


class DbReplayTable(Base):
    __tablename__ = "replay_data"

    use_extra_info = False

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    message_type = Column(String(64))

    # specify the pickler to allow backwards compatibility between 3.7 and 3.8
    message_contents = Column(PickleType(pickler=pickle))

    extra_info = Column(PickleType(pickler=pickle))

    master_table_id = Column(Integer, ForeignKey("master.unique_id"))
    parent = relationship("DBMasterTable", back_populates="children_replay")

    __mapper_args__ = {}

    @classmethod
    def from_sqlite(cls, row):
        this = DbReplayTable()
        this.unique_id = row["unique_id"]
        this.timestamp = row["timestamp"]
        this.message_type = row["message_type"]
        this.message_contents = row["message_contents"]
        this.master_table_id = row["master_table_id"]

        if "extra_info" in row:
            this.extra_info = row["extra_info"]
        else:
            this.extra_info = None

        this.strat = row["strat"]
        return this

    def __repr__(self):
        return (
            f"<DbReplayTable(unique_id={self.unique_id})"
            f", timestamp={self.timestamp}, "
            f"message_type={self.message_type}"
            f", master_table_id={self.master_table_id})>"
        )

    @staticmethod
    def _has_extra_info(engine):
        result = engine.execute(
            "SELECT COUNT(*) FROM pragma_table_info('replay_data') WHERE name='extra_info'"
        )
        rows = result.fetchall()
        count = rows[0][0]
        return count != 0

    @staticmethod
    def _configs_require_conversion(engine):
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        results = session.query(DbReplayTable).all()

        for result in results:
            if result.message_contents["type"] == "setup":
                config_str = result.message_contents["message"]["config_str"]
                config = Config(config_str=config_str)
                if config.version < __version__:
                    return True  # assume that if any config needs to be refactored, all of them do

        return False

    @staticmethod
    def update(engine):
        logger.info("DbReplayTable : update called")

        if not DbReplayTable._has_extra_info(engine):
            DbReplayTable._add_extra_info(engine)

        if DbReplayTable._configs_require_conversion(engine):
            DbReplayTable._convert_configs(engine)

    @staticmethod
    def requires_update(engine):
        return not DbReplayTable._has_extra_info(
            engine
        ) or DbReplayTable._configs_require_conversion(engine)

    @staticmethod
    def _add_extra_info(engine):
        try:
            result = engine.execute(
                "SELECT COUNT(*) FROM pragma_table_info('replay_data') WHERE name='extra_info'"
            )
            rows = result.fetchall()
            count = rows[0][0]

            if 0 == count:
                logger.debug(
                    "Altering the replay_data table to add the extra_info column"
                )
                engine.execute("ALTER TABLE replay_data ADD COLUMN extra_info BLOB")
                engine.commit()
        except Exception as e:
            logger.debug(f"Column already exists, no need to alter. [{e}]")

    @staticmethod
    def _convert_configs(engine):
        Session = sessionmaker(bind=engine)
        session = Session()
        results = session.query(DbReplayTable).all()

        for result in results:
            if result.message_contents["type"] == "setup":
                config_str = result.message_contents["message"]["config_str"]
                config = Config(config_str=config_str)
                if config.version < __version__:
                    config.convert_to_latest()
                new_str = str(config)

                new_message = {"type": "setup", "message": {"config_str": new_str}}
                if "version" in result.message_contents:
                    new_message["version"] = result.message_contents["version"]

                result.message_contents = new_message

        session.commit()
        logger.info("DbReplayTable : updated old configs.")


class DbStratTable(Base):
    __tablename__ = "strat_data"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    strat = Column(PickleType(pickler=pickle))

    master_table_id = Column(Integer, ForeignKey("master.unique_id"))
    parent = relationship("DBMasterTable", back_populates="children_strat")

    @classmethod
    def from_sqlite(cls, row):
        this = DbStratTable()
        this.unique_id = row["unique_id"]
        this.timestamp = row["timestamp"]
        this.strat = row["strat"]
        this.master_table_id = row["master_table_id"]

        return this

    def __repr__(self):
        return (
            f"<DbStratTable(unique_id={self.unique_id})"
            f", timestamp={self.timestamp} "
            f", master_table_id={self.master_table_id})>"
        )

    @staticmethod
    def update(engine):
        logger.info("DbStratTable : update called")

    @staticmethod
    def requires_update(engine):
        return False


class DbConfigTable(Base):
    __tablename__ = "config_data"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    config = Column(PickleType(pickler=pickle))

    master_table_id = Column(Integer, ForeignKey("master.unique_id"))
    parent = relationship("DBMasterTable", back_populates="children_config")

    @classmethod
    def from_sqlite(cls, row):
        this = DbConfigTable()
        this.unique_id = row["unique_id"]
        this.timestamp = row["timestamp"]
        this.strat = row["config"]
        this.master_table_id = row["master_table_id"]

        return this

    def __repr__(self):
        return (
            f"<DbStratTable(unique_id={self.unique_id})"
            f", timestamp={self.timestamp} "
            f", master_table_id={self.master_table_id})>"
        )

    @staticmethod
    def update(engine):
        logger.info("DbConfigTable : update called")

    @staticmethod
    def requires_update(engine):
        return False


class DbRawTable(Base):
    """
    Fact table to store the raw data of each iteration of an experiment.
    """

    __tablename__ = "raw_data"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    model_data = Column(Boolean)

    master_table_id = Column(Integer, ForeignKey("master.unique_id"))
    parent = relationship("DBMasterTable", back_populates="children_raw")
    children_param = relationship("DbParamTable", back_populates="parent")
    children_outcome = relationship("DbOutcomeTable", back_populates="parent")

    @classmethod
    def from_sqlite(cls, row):
        this = DbRawTable()
        this.unique_id = row["unique_id"]
        this.timestamp = row["timestamp"]
        this.model_data = row["model_data"]
        this.master_table_id = row["master_table_id"]

        return this

    def __repr__(self):
        return (
            f"<DbRawTable(unique_id={self.unique_id})"
            f", timestamp={self.timestamp} "
            f", master_table_id={self.master_table_id})>"
        )

    @staticmethod
    def update(db, engine):
        logger.info("DbRawTable : update called")

        # Get every master table
        for master_table in db.get_master_records():

            # Get raw tab
            for message in master_table.children_replay:
                if message.message_type != "tell":
                    continue

                timestamp = message.timestamp

                # Deserialize pickle message
                message_contents = message.message_contents

                # Get outcome
                outcomes = message_contents["message"]["outcome"]
                # Get parameters
                params = message_contents["message"]["config"]
                # Get model_data
                model_data = message_contents["message"].get("model_data", True)

                db_raw_record = db.record_raw(
                    master_table=master_table,
                    model_data=bool(model_data),
                    timestamp=timestamp,
                )

                for param_name, param_value in params.items():
                    if isinstance(param_value, Iterable) and type(param_value) != str:
                        if len(param_value) == 1:
                            db.record_param(
                                raw_table=db_raw_record,
                                param_name=str(param_name),
                                param_value=float(param_value[0]),
                            )
                        else:
                            for j, v in enumerate(param_value):
                                db.record_param(
                                    raw_table=db_raw_record,
                                    param_name=str(param_name) + "_stimuli" + str(j),
                                    param_value=float(v),
                                )
                    else:
                        db.record_param(
                            raw_table=db_raw_record,
                            param_name=str(param_name),
                            param_value=float(param_value),
                        )

                if isinstance(outcomes, Iterable) and type(outcomes) != str:
                    for j, outcome_value in enumerate(outcomes):
                        if (
                            isinstance(outcome_value, Iterable)
                            and type(outcome_value) != str
                        ):
                            if len(outcome_value) == 1:
                                outcome_value = outcome_value[0]
                            else:
                                raise ValueError(
                                    "Multi-outcome values must be a list of lists of length 1!"
                                )
                        db.record_outcome(
                            raw_table=db_raw_record,
                            outcome_name="outcome_" + str(j),
                            outcome_value=float(outcome_value),
                        )
                else:
                    db.record_outcome(
                        raw_table=db_raw_record,
                        outcome_name="outcome",
                        outcome_value=float(outcomes),
                    )

    @staticmethod
    def requires_update(engine):
        """Check if the raw table is empty, and data already exists."""
        n_raws = engine.execute("SELECT COUNT (*) FROM raw_data").fetchone()[0]
        n_tells = engine.execute(
            "SELECT COUNT (*) FROM replay_data \
            WHERE message_type = 'tell'"
        ).fetchone()[0]

        if n_raws == 0 and n_tells != 0:
            return True
        return False


class DbParamTable(Base):
    """
    Dimension table to store the parameters of each iteration of an experiment.
    Supports multiple parameters per iteration, and multiple stimuli per parameter.
    """

    __tablename__ = "param_data"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    param_name = Column(String(50))
    param_value = Column(String(50))

    iteration_id = Column(Integer, ForeignKey("raw_data.unique_id"))
    parent = relationship("DbRawTable", back_populates="children_param")

    @classmethod
    def from_sqlite(cls, row):
        this = DbParamTable()
        this.unique_id = row["unique_id"]
        this.param_name = row["param_name"]
        this.param_value = row["param_value"]
        this.iteration_id = row["iteration_id"]

        return this

    def __repr__(self):
        return (
            f"<DbParamTable(unique_id={self.unique_id})"
            f", iteration_id={self.iteration_id}>"
        )

    @staticmethod
    def update(engine):
        logger.info("DbParamTable : update called")

    @staticmethod
    def requires_update(engine):
        return False


class DbOutcomeTable(Base):
    """
    Dimension table to store the outcomes of each iteration of an experiment.
    Supports multiple outcomes per iteration.
    """

    __tablename__ = "outcome_data"

    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    outcome_name = Column(String(50))
    outcome_value = Column(Float)

    iteration_id = Column(Integer, ForeignKey("raw_data.unique_id"))
    parent = relationship("DbRawTable", back_populates="children_outcome")

    @classmethod
    def from_sqlite(cls, row):
        this = DbOutcomeTable()
        this.unique_id = row["unique_id"]
        this.outcome_name = row["outcome_name"]
        this.outcome_value = row["outcome_value"]
        this.iteration_id = row["iteration_id"]

        return this

    def __repr__(self):
        return (
            f"<DbOutcomeTable(unique_id={self.unique_id})"
            f", iteration_id={self.iteration_id}>"
        )

    @staticmethod
    def update(engine):
        logger.info("DbOutcomeTable : update called")

    @staticmethod
    def requires_update(engine):
        return False
