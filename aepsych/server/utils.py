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
from aepsych.database.utils import combine_dbs

logger = utils_logging.getLogger(logging.INFO)


def get_next_filename(folder, fname, ext):
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n + 1}.{ext}"


def parse_argument(args):
    parser = argparse.ArgumentParser(description="AEPsych Database!")

    parser.add_argument(
        "-d",
        "--db",
        type=str,
        help="The database to use if not the default (./databases/default.db).",
        default=None,
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize the data contained in a database.",
    )

    parser.add_argument(
        "--tocsv",
        type=str,
        help="Export the data to a csv file with the provided path.",
    )

    parser.add_argument(
        "--combine",
        type=str,
        help="Combine csvs within the listed directory into a single db at the -db path.",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclude certain files for other commands, this is currently only used for combine to exclude databases.",
        action="extend",
        nargs="*",
        default=[],
    )

    return parser.parse_args(args)


def run_database(args):
    logger.info("Starting AEPsych Database!")
    try:
        database_path = args.db
        read_only = args.summarize or "tocsv" in args and args.tocsv is not None

        if "combine" in args and args.combine is not None:
            n_exp = combine_dbs(
                out_path=database_path,
                dbs=args.combine,
                exclude=args.exclude,
            )
            logger.info(f"Combined {n_exp} experiment sessions into {database_path}")
            return

        database = db.Database(database_path, read_only=read_only)

        if args.summarize:
            summary = database.summarize_experiments()
            print(summary)
            database.cleanup()

        elif "tocsv" in args and args.tocsv is not None:
            try:
                database.to_csv(args.tocsv)
                logger.info(f"Exported contents of {database_path} to {args.tocsv}")
            except Exception as error:
                logger.error(
                    f"Failed to export contents of {database_path} with error `{error}`"
                )
            database.cleanup()

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
    args = parse_argument(sys.argv[1:])
    run_database(args)


if __name__ == "__main__":
    main()
