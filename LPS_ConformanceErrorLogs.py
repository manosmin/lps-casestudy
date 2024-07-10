import pandas as pd
import pm4py

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

    LPS_dir = 'LPS_Logs'
    
    # Create LPS log (Type A)
    LPS_log_A = createLPSLog(f'{LPS_dir}/LiqueurPlant2024_LOG.txt', True)

    # Discover petri net (Alpha Miner)
    net1, im1, fm1 = pm4py.discover_petri_net_alpha(LPS_log_A)

    # Discover petri net (Inductive Miner)
    net2, im2, fm2 = pm4py.discover_petri_net_inductive(LPS_log_A)

    # Discover petri net (Heuristics Miner)
    net3, im3, fm3 = pm4py.discover_petri_net_heuristics(LPS_log_A)

    ### Create LPS logs with errors ###

    # 1: Missing activities
    error_logA1 = createLPSLog(f'{LPS_dir}/A_ERROR_LOG1.txt', True)

    # 2: Wrong order activities
    error_logA2 = createLPSLog(f'{LPS_dir}/A_ERROR_LOG2.txt', True)

    # 3: Duplicate activities
    error_logA3 = createLPSLog(f'{LPS_dir}/A_ERROR_LOG3.txt', True)

    error_logsA = [error_logA1, error_logA2, error_logA3]

    for errorLog in error_logsA:

        # Check error log conformance against petri net (Alpha Miner)
        print(pm4py.precision_token_based_replay(errorLog, net1, im1, fm1))
        print(pm4py.fitness_token_based_replay(errorLog, net1, im1, fm1))
        try: 
            print(pm4py.fitness_alignments(errorLog, net1, im1, fm1)) 
        except Exception as e: 
            print(e)

        # Check error log conformance against petri net (Inductive Miner)
        print(pm4py.precision_token_based_replay(errorLog, net2, im2, fm2))
        print(pm4py.fitness_token_based_replay(errorLog, net2, im2, fm2))
        try: 
            print(pm4py.fitness_alignments(errorLog, net2, im2, fm2)) 
        except Exception as e: 
            print(e)

        # Check error log conformance against petri net (Heuristics Miner)
        print(pm4py.precision_token_based_replay(errorLog, net3, im3, fm3))
        print(pm4py.fitness_token_based_replay(errorLog, net3, im3, fm3))
        try: 
            print(pm4py.fitness_alignments(errorLog, net3, im3, fm3)) 
        except Exception as e: 
            print(e)

    # Create LPS log (Type B)
    LPS_log_B = createLPSLog(f'{LPS_dir}/LiqueurPlant2024_LOG.txt', False)

    # Discover petri net (Alpha Miner)
    net1, im1, fm1 = pm4py.discover_petri_net_alpha(LPS_log_B)

    # Discover petri net (Inductive Miner)
    net2, im2, fm2 = pm4py.discover_petri_net_inductive(LPS_log_B)

    # Discover petri net (Heuristics Miner)
    net3, im3, fm3 = pm4py.discover_petri_net_heuristics(LPS_log_B)

    ### Create LPS logs with errors ###

    # 1: Missing activities
    error_logB1 = createLPSLog(f'{LPS_dir}/B_ERROR_LOG1.txt', False)

    # 2: Wrong order activities
    error_logB2 = createLPSLog(f'{LPS_dir}/B_ERROR_LOG2.txt', False)

    # 3: Duplicate activities
    error_logB3 = createLPSLog(f'{LPS_dir}/B_ERROR_LOG3.txt', False)

    error_logsB = [error_logB1, error_logB2, error_logB3]

    for errorLog in error_logsB:
        # Check error log conformance against petri net (Alpha Miner)
        print(pm4py.precision_token_based_replay(errorLog, net1, im1, fm1))
        print(pm4py.fitness_token_based_replay(errorLog, net1, im1, fm1))
        try: 
            print(pm4py.fitness_alignments(errorLog, net1, im1, fm1)) 
        except Exception as e: 
            print(e)

        # Check error log conformance against petri net (Inductive Miner)
        print(pm4py.precision_token_based_replay(errorLog, net2, im2, fm2))
        print(pm4py.fitness_token_based_replay(errorLog, net2, im2, fm2))
        try: 
            print(pm4py.fitness_alignments(errorLog, net2, im2, fm2)) 
        except Exception as e: 
            print(e)

        # Check error log conformance against petri net (Heuristics Miner)
        print(pm4py.precision_token_based_replay(errorLog, net3, im3, fm3))
        print(pm4py.fitness_token_based_replay(errorLog, net3, im3, fm3))
        try: 
            print(pm4py.fitness_alignments(errorLog, net3, im3, fm3)) 
        except Exception as e: 
            print(e)