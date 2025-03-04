#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def handle_finish_strategy(self, request):
    self.strat.finish()
    return {
        "finished_strategy": self.strat.name,
        "finished_strat_idx": self.strat._strat_idx,
    }
