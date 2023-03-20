#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys

import aepsych.database.db as db
import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def get_next_filename(folder, fname, ext):
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n+1}.{ext}"


def parse_argument():
    parser = argparse.ArgumentParser(description="AEPsych Database!")

    parser.add_argument(
        "-l",
        "--list",
        help="Lists available experiments in the database.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--db",
        type=str,
        help="The database to use if not the default (./databases/default.db).",
        default=None,
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update the database tables with the most recent columns and tables.",
    )

    args = parser.parse_args()
    return args


def run_database(args):
    logger.info("Starting AEPsych Database!")
    try:
        database_path = args.db
        database = db.Database(database_path)
        if args.list is True:
            database.list_master_records()
        elif "update" in args and args.update:
            logger.info(f"Updating the database {database_path}")
            if database.is_update_required():
                database.perform_updates()
                logger.info(f"- updated database {database_path}")
            else:
                logger.info(f"- update not needed for database {database_path}")

    except (KeyboardInterrupt, SystemExit):
        logger.exception("Got Ctrl+C, exiting!")
        sys.exit()
    except RuntimeError as e:
        fname = get_next_filename(".", "dump", "pkl")
        logger.exception(f"CRASHING!! dump in {fname}")
        raise RuntimeError(e)


def main():
    args = parse_argument()
    run_database(args)


if __name__ == "__main__":
    main()
