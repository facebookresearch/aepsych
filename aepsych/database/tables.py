#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import pickle

from sqlalchemy import (
    Column,
    ForeignKey,
    String,
    Integer,
    DateTime,
    PickleType,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from aepsych.config import Config

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

    children_replay = relationship("DbReplayTable", back_populates="parent")
    children_strat = relationship("DbStratTable", back_populates="parent")
    children_config = relationship("DbConfigTable", back_populates="parent")
    """
    @classmethod
    def from_sqlite(cls, row):
        this = DBMasterTable()
        this.unique_id = row["unique_id"]
        this.experiment_name = row["experiment_name"]
        this.experiment_description = row["experiment_description"]
        this.experiment_id = row["experiment_id"]
        return this
    """

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

    @staticmethod
    def requires_update(engine):
        return False


# link back to the master table entry
#
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
                if config.version == "0.0":
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
                if config.version == "0.0":
                    config.convert("0.0", "0.1")
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
