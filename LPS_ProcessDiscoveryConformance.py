import pandas as pd
import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristic_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

A_ACTIVITIES = {'s1-StartFilling', 's1-FillingCompleted', 'lgpA-StartTransfer', 's1-PouringCompleted', 'lgpA-TransferCompleted', 's4-FillingCompleted', 's4-PouringStarted', 's4-PouringCompleted', 's4-StartMixing', 's4-MixingCompleted', 's4-StartHeating', 's4-HeatingCompleted'}

def createLPSLog(filename, option):
    df = pd.read_csv(filename, sep=',')
    df['activity'] = df['source'] + '-' + df['info']
    df = df.drop(['source', 'info'], axis=1)
    df = df.rename(columns={'case': 'case:concept:name', 'activity': 'concept:name', 'timestamp': 'time:timestamp'})
    df = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

    new_log = pm4py.convert_to_event_log(df)
    new_log = pm4py.filter_event_attribute_values(new_log, attribute_key = 'concept:name', values = A_ACTIVITIES, level = 'event', retain = option) 

    return new_log

if __name__ == '__main__':

    # Create LPS log (Type A)
    LPS_log_A = createLPSLog('LiqueurPlant2024_LOG.txt', True)

    # Create LPS log (Type B)
    LPS_log_B = createLPSLog('LiqueurPlant2024_LOG.txt', False)

    LPS_logs = [LPS_log_A, LPS_log_B]

    for log in LPS_logs:

        # Discover Heuristic Net
        heu_net = heuristic_miner.apply_heu(log)
        gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.view(gviz)
        '''hn_visualizer.save(gviz, 'HeuristicNet')'''

        # Discover petri net (Alpha Miner)
        net1, im1, fm1 = pm4py.discover_petri_net_alpha(log)
        gviz = pn_visualizer.apply(net1, im1, fm1)
        pn_visualizer.view(gviz)
        '''pn_visualizer.save(gviz, 'AlphaMiner')'''

        # Check log conformance against petri net (Alpha Miner)
        print(pm4py.precision_token_based_replay(log, net1, im1, fm1))
        print(pm4py.fitness_token_based_replay(log, net1, im1, fm1))
        try: 
            print(pm4py.fitness_alignments(log, net1, im1, fm1)) 
        except Exception as e: 
            print(e)
        
        # Discover petri net (Inductive Miner)
        net2, im2, fm2 = pm4py.discover_petri_net_inductive(log)
        gviz = pn_visualizer.apply(net2, im2, fm2)
        pn_visualizer.view(gviz)
        '''pn_visualizer.save(gviz, 'InductiveMiner')'''

        # Check log conformance against petri net (Inductive Miner)
        print(pm4py.precision_token_based_replay(log, net2, im2, fm2))
        print(pm4py.fitness_token_based_replay(log, net2, im2, fm2))
        try: 
            print(pm4py.fitness_alignments(log, net2, im2, fm2)) 
        except Exception as e: 
            print(e)

        # Discover petri net (Heuristics Miner)
        net3, im3, fm3 = pm4py.discover_petri_net_heuristics(log)
        gviz = pn_visualizer.apply(net3, im3, fm3)
        pn_visualizer.view(gviz)
        '''pn_visualizer.save(gviz, 'HeuristicsMiner')'''

        # Check log conformance against petri net (Heuristics Miner)
        print(pm4py.precision_token_based_replay(log, net3, im3, fm3))
        print(pm4py.fitness_token_based_replay(log, net3, im3, fm3))
        try: 
            print(pm4py.fitness_alignments(log, net3, im3, fm3)) 
        except Exception as e: 
            print(e)
