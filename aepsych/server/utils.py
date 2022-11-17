#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

class AEPsychUtils(object):
    pass

def parse_argument():
    parser = argparse.ArgumentParser(description="AEPsych Utils!")

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
        "-r", "--replay", type=str, help="UUID of the experiment to replay."
    )

    parser.add_argument(
        "-m", "--resume", action="store_true", help="Resume server after replay."
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update the database tables with the most recent columns.",
    )

if __name__ == "__main__":
    print('Hi!')