import pandas as pd


class BenchmarkLogger:
    def __init__(self, log_every=None):
        self._log = []
        self.log_every = log_every

    def log_at(self, i):
        if self.log_every is not None:
            return i % self.log_every == 0
        else:
            return False

    def log(self, strat, flatconfig, metrics, trial_id, elapsed, rep, final=False):
        out = {"elapsed": elapsed, "trial_id": trial_id, "rep": rep, "final": final}
        out.update(flatconfig)
        out.update(metrics)
        self._log.append(out)

    def pandas(self):
        return pd.DataFrame(self._log)
