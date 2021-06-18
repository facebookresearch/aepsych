try:
    from psychopy.data import StairHandler

except ImportError:

    # stub it out
    import warnings

    warnings.warn("psychopy not installed, stubbing out StairHandler compatibility!")

    class StairHandler:
        def __init__(self, startVal, nTrials, stepType):
            self.startVal = startVal
            self.nTrials = nTrials
            self.stepType = stepType


import pandas as pd
from aepsych.config import Config
from aepsych.server import AEPsychServer


class DummySocket(object):
    pass


class AEPsychHandler(StairHandler):
    def __init__(self, config_path, database_name=None, epsilon_prob=0.1):

        self.server = AEPsychServer(socket=DummySocket(), database_path=database_name)

        with open(config_path) as f:
            config_str = f.read()
        message = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        self.server.versioned_handler(message)

        self.dim = self.server.strat.dim

        self.stimulus = self.server.parnames
        self.epsilon_prob = epsilon_prob
        StairHandler.__init__(
            self, startVal=None, nTrials=self.server.strat.n_trials, stepType=None
        )

        startVal = self.server.unversioned_handler({"type": "ask", "message": ""})
        self.finished = False
        self.startVal = startVal
        if self.startIntensity is not None:
            self._nextIntensity = self.startIntensity
        # self._nextIntensity = startVal

    @property
    def startIntensity(self):
        return self.startVal

    def __iter__(self):
        return self

    def addResponse(self, result, reaction_time=None, intensity=None):
        """Add a 1 or 0 to signify a correct / detected or
        incorrect / missed trial

        Supplying an `intensity` value here indicates that you did not use the
        recommended intensity in your last trial and the staircase will
        replace its recorded value with the one you supplied here.
        """
        # Process user supplied intensity
        if intensity is None:
            intensity = self._nextIntensity
        else:
            # Update the intensity.
            if len(self.intensities) != 0:
                self.intensities.pop()  # remove the auto-generated one
            self.intensities.append(intensity)
        # Update aepsych server
        request = {
            "type": "tell",
            "message": {"config": intensity, "outcome": result},  # 0 or 1
            "extra_info": {"reaction_time": reaction_time},
        }
        self.server.unversioned_handler(request)
        # Update other things
        self.data.append(result)
        # add the current data to experiment if poss
        if self.getExp() is not None:  # update the experiment handler too
            self.getExp().addData(self.name + ".response", result)

        self._checkFinished()
        if not self.finished:
            self.calculateNextIntensity()

    def calculateNextIntensity(self):
        """assigns the next intensity level"""
        self._nextIntensity = self.server.unversioned_handler(
            {"type": "ask", "message": ""}
        )

    def __next__(self):
        """Advances to next trial and returns it.
        Updates attributes; `thisTrial`, `thisTrialN`, `thisIndex`,
        `finished`, `intensities`

        If the trials have ended, calling this method will raise a
        StopIteration error. This can be handled with code such as::

            staircase = data.QuestHandler(.......)
            for eachTrial in staircase:  # automatically stops when done
                # do stuff

        or::

            staircase = data.QuestHandler(.......)
            while True:  # i.e. forever
                try:
                    thisTrial = staircase.next()
                except StopIteration:  # we got a StopIteration error
                    break  # break out of the forever loop
                # do stuff here for the trial
        """
        if not self.finished:
            # update pointer for next trial
            self.thisTrialN += 1
            self.intensities.append(self._nextIntensity)
            return self._nextIntensity
        else:
            self._terminate()

    next = __next__  # allows user to call without a loop `val = trials.next()`

    def _checkFinished(self):
        """checks if we are finished
        Updates attribute: `finished`
        """
        self.finished = self.server.strat.finished

    def to_pd(self):

        exp_ids = [rec.experiment_id for rec in self.server.db.get_master_records()]
        dfs = []
        for exp_id in exp_ids:
            dfs.append(self.server.get_dataframe_from_replay(exp_id))
        datie = pd.concat(dfs)

        return datie

    @classmethod
    def from_config(cls, config_message):
        config = Config()
        config.update(
            config_fnames=[config_message.message_contents["message"]["config_path"]]
        )
        names = config.str_to_list(
            config.get("experiment", "parnames"), element_type=str
        )
        outcome_type = config.get("experiment", "outcome_type")
        return (names, outcome_type)
