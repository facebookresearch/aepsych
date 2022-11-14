#!/usr/bin/env python3
# coding: utf-8

# In[ ]:


from aepsych.server import AEPsychServer
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
db_name = "../databases/default.db"
outputfile = '../databases/results.csv'


# In[ ]:


class DummySocket(object):
    pass

serv = AEPsychServer(socket=DummySocket, database_path=db_name)

exp_ids = [rec.experiment_id for rec in serv.db.get_master_records()]


# In[ ]:


# this will take all the data a write it to a csv
def from_setup(setup_message):
    confs = setup_message.message_contents['message']['experiment_config']
    names = []
    for con in confs:
        names.append(con['name'])
    outcome_type = setup_message.message_contents['message']['outcome_type']
    return(names, outcome_type)

def get_data(serv, exp_id):
    recs = serv.db.get_replay_for(exp_id)
    names, outcome_type = from_setup(recs[0])
    results = defaultdict(list)
    for rec in recs:
        if rec.message_type == "tell":
            for name in names:
                results[name].append(rec.message_contents['message']['config'][name])
                results['outcome'].append(rec.message_contents['message']['outcome'])
    datie = pd.DataFrame(results['squeeze']).add_prefix('squeeze_')
    datie['outcome'] = results['outcome']
    datie['exp_id'] = exp_id
    return(datie)

dfs = []
for exp_id in exp_ids:
    dfs.append(get_data(serv, exp_id))
datie = pd.concat(dfs)

datie.to_csv(outputfile)


# In[ ]:


from aepsych.plotting import make_debug_plots
# do the replays and produce plots
for exp_id in exp_ids:
    serv.replay(exp_id)
    make_debug_plots(serv.strat)

