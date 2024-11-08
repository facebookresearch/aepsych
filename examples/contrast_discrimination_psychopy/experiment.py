#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import experiment_config
import numpy as np
import torch
from aepsych_client import AEPsychClient
from helpers import HalfGrating
from psychopy import core, data, event, gui, monitors, visual
from psychopy.tools.filetools import toFile


def run_experiment():
    seed = experiment_config.constants["seed"]
    config_path = experiment_config.constants["config_path"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    expInfo = {"observer": "default_observer"}
    expInfo["dateStr"] = data.getDateStr()  # add the current time

    # present a dialogue to change params
    dlg = gui.DlgFromDict(expInfo, title="multi-D JND Exp", fixed=["dateStr"])
    if dlg.OK:
        toFile("lastParams.pickle", expInfo)  # save params to file for next time
    else:
        core.quit()  # the user hit cancel so exit

    screen = monitors.Monitor("testMonitor", gamma=1)

    screen.setSizePix(experiment_config.psychopy_vars["setSizePix"])
    screen.setWidth(experiment_config.psychopy_vars["setWidth"])
    screen.setDistance(experiment_config.psychopy_vars["setDistance"])

    win = visual.Window(
        allowGUI=True,
        units="deg",
        monitor=screen,
        bpc=(8, 8, 8),
        size=experiment_config.psychopy_vars["setSizePix"],
        fullscr=False,
    )
    screen_text_g = visual.TextStim(win, text=None, alignHoriz="center", color="green")
    screen_text_r = visual.TextStim(win, text=None, alignHoriz="center", color="red")
    screen_text = visual.TextStim(win, text=None, alignHoriz="center", color="gray")

    # display instructions and wait
    message2 = visual.TextStim(
        win,
        pos=[0, +3],
        text="Hit the space bar key when ready and "
        "to advance to the next trial after you see a red cross.",
    )
    message1 = visual.TextStim(
        win,
        pos=[0, -3],
        text="You'll see a stimulus. One side will have a grating and the other will be noise."
        "  "
        "Press left or right corresponding to the side with noise. If you don't know, please guess.",
    )
    message1.draw()
    message2.draw()
    win.flip()  # to show our newly drawn 'stimuli'
    # pause until there's a keypress
    event.waitKeys()

    # start the trial: draw grating
    clock = core.Clock()

    screen_text_r.setText("+")
    screen_text_r.draw(win=win)
    win.flip()

    aepsych_client = AEPsychClient()
    aepsych_client.configure(config_path=config_path)

    # create stimulus
    stim = HalfGrating(**experiment_config.base_params, win=win)
    i = 0
    is_finished = False
    while not is_finished:
        ask_response = aepsych_client.ask()
        trial_params = ask_response["config"]
        is_finished = ask_response["is_finished"]
        stim.update(trial_params)
        print(trial_params)

        bg_color = np.array([stim.pedestal_psychopy_scale] * 3)
        win.setColor(bg_color)
        win.color = bg_color
        win.flip()

        screen_text_r.setText("+")
        screen_text_r.draw(win=win)
        win.flip()

        core.wait(experiment_config.psychopy_vars["iti"])

        fixation_keys = []
        while not fixation_keys:
            fixation_keys = event.getKeys(keyList=["space"])

        fixation_keys = ["space"]  ## for debugging

        if "space" in fixation_keys:
            screen_text.setText("+")
            screen_text.draw(win=win)
            win.flip()

            noisy_half = "left" if np.random.randint(2) == 0 else "right"
            clock.reset()

            keys = stim.draw(
                noisy_half=noisy_half,
                win=win,
                pre_duration_s=experiment_config.psychopy_vars["pre_duration_s"],
                stim_duration_s=experiment_config.psychopy_vars["stim_duration_s"],
            )
            # keys = event.waitKeys(keyList=["left", "right"])  # phil took out max wait
            rt = clock.getTime()
            response = noisy_half in keys
            print(f"keys:{keys}, ca:{noisy_half}, acc:{response}, rt:{rt}")
            win.flip()

            if response:
                screen_text_g.setText("Correct")
                screen_text_g.draw()
                win.flip()
            else:
                screen_text_r.setText("Incorrect")
                screen_text_r.draw()
                win.flip()

            # inform bayesopt of the response, needed to calculate next contrast

            aepsych_client.tell(config=trial_params, outcome=response, rt=rt)
            # core.wait(experiment_config.psychopy_vars["post_duration_s"])
            event.clearEvents()
            print(f"trial {i}")
            i = i + 1

    win.close()

    aepsych_client.finalize()
    core.quit()


if __name__ == "__main__":
    run_experiment()
