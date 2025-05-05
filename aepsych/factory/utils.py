#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from contextlib import contextmanager

__default_invgamma_concentration = 4.6
__default_invgamma_rate = 1.0
DEFAULT_INVGAMMA_CONC = 4.6
DEFAULT_INVGAMMA_RATE = 1.0


@contextmanager
def temporary_attributes(obj, **kwargs):
    """Temporarily sets attributes on an object, and restores them when the context exits."""

    try:
        old_attrs = {}
        for attr, val in kwargs.items():
            old_attrs[attr] = getattr(obj, attr)
            setattr(obj, attr, val)
        yield obj
    finally:
        for attr, val in old_attrs.items():
            setattr(obj, attr, val)
