#!/usr/bin/env python
# coding: utf-8


## usage: texp_cut_and_batch.py [/path/to/input/directory/with/texp/files/] [/path/to/output/directory/] [number of input batches] [number of export batches] [band]

# import pandas as pd
# import numpy as np
# import os, sys, configparser
# import argparse

# def texp_cut_and_batch(texp_dir,out_dir,N_in,N_out,band,min_texp):
#     '''
#     Function to:
#         1. Load in and re-aggregate the results of a batched run of max-texp-by-sky-loc.py
#         2. Ensure all exposure times are at least min_texp s.
#         3. Rebatch and create files for use by the scheduler.

#     Arguments
#     -----------------------
#     texp_dir (str)   : /path/to/directory with downselected event csvs
#     out_dir (str)    : /path/to/save/directory
#     N_in (int)       : Number of input batches (as determined by max-texp-by-sky-loc.py)
#     N_out (int)      : Number of output batches (for the scheduler)
#     band (str)       : Which UV band is being considered ('nuv' or 'fuv'). Just used in this script to parse filenames.
#     min_texp (float) : Minimum allowed exposure time in seconds. Exposure times <= min_texp will be set to this value.
#     '''

#     ## load downselected (< max_area sq. deg localization, < max_texp t_exp) events
#     events = pd.read_csv(texp_dir+'allsky_texp_max_cut_'+band+'_batch0.txt',delimiter=' ')
#     for i in range(1,N_in):
#         next_batch = pd.read_csv(texp_dir+'allsky_texp_max_cut_'+band+'_batch'+str(i)+'.txt',delimiter=' ')
#         events = pd.concat([events, next_batch], ignore_index=True)

#     ## liaise calculated t_exp to desired scheduler t_exp
#     ## i.e., ensure minimum 500s exposures, convert s to ks, reformat for use by the scheduler
#     texp_sched = events['texp_max (s)'].to_numpy(copy=True)
#     for i, (t, ev_id) in enumerate(zip(texp_sched,events['event_id'].to_list())):
#         if t <= min_texp:
#             texp_sched[i] = min_texp
#         else:
#             continue
#     texp_sched = texp_sched/1000
#     events_texp = pd.DataFrame({'event_id':events['event_id'].tolist(),'t_exp (ks)':texp_sched})

#     events_texp.to_csv(os.path.join(out_dir,'allsky_sched_full.txt'),index=False,sep=' ')

#     ## batch
#     batch_dir = os.path.join(out_dir,'texp_sched')
#     os.makedirs(batch_dir, exist_ok=True) ## weizmann
#     list_of_lists = np.array_split(events_texp,N_out)
#     batchnums = range(len(list_of_lists))
#     for lst, num in zip(list_of_lists,batchnums):
#         filename = os.path.join(batch_dir,'allsky_sched_batch'+str(num)+'.txt')
#         lst.to_csv(filename,index=False,sep=',')

#     return


# if __name__ == '__main__':

#     ## set up argparser
#     parser = argparse.ArgumentParser(description="Re-aggregate the output from max-texp-by-sky-loc.py and prepare it for passing to the UVEX scheduler.")
#     parser.add_argument('params', type=str, help='/path/to/params_file.ini')

#     args = parser.parse_args()

#     ## set up configparser
#     config = configparser.ConfigParser()
#     config.read(args.params)

#     ## get info from params file
#     out_dir          = config.get("params","save_directory")
#     N_in             = int(config.get("params","N_batch_preproc",fallback=1))
#     N_out            = int(config.get("params","N_batch_sched",fallback=N_in))
#     band             = config.get("params","band")
#     min_texp         = float(config.get("params","min_texp"))

#     texp_dir = out_dir+'/texp_out/'

#     os.makedirs(texp_dir , exist_ok=True)  ## weizmann

#     ## run the script
#     texp_cut_and_batch(texp_dir,out_dir,N_in,N_out,band,min_texp)


import pandas as pd
import numpy as np
import os, sys, configparser
import argparse


def texp_cut_and_batch(texp_dir, out_dir, N_in, N_out, band, min_texp):
    """
    Function to:
        1. Load in and re-aggregate the results of a batched run of max-texp-by-sky-loc.py
        2. Ensure all exposure times are at least min_texp s.
        3. Rebatch and create files for use by the scheduler.

    Arguments
    -----------------------
    texp_dir (str)   : /path/to/directory with downselected event csvs
    out_dir (str)    : /path/to/save/directory
    N_in (int)       : Number of input batches (as determined by max-texp-by-sky-loc.py)
    N_out (int)      : Number of output batches (for the scheduler)
    band (str)       : Which UV band is being considered ('nuv' or 'fuv'). Just used in this script to parse filenames.
    min_texp (float) : Minimum allowed exposure time in seconds. Exposure times <= min_texp will be set to this value.
    """

    ## load downselected (< max_area sq. deg localization, < max_texp t_exp) events
    events = pd.read_csv(
        texp_dir + "allsky_texp_max_cut_" + band + "_batch0.txt", delimiter=" "
    )
    for i in range(1, N_in):
        next_batch = pd.read_csv(
            texp_dir + "allsky_texp_max_cut_" + band + "_batch" + str(i) + ".txt",
            delimiter=" ",
        )
        events = pd.concat([events, next_batch], ignore_index=True)

    ## liaise calculated t_exp to desired scheduler t_exp
    ## i.e., ensure minimum 500s exposures, convert s to ks, reformat for use by the scheduler
    texp_sched = events["texp_max (s)"].to_numpy(copy=True)
    for i, (t, ev_id) in enumerate(zip(texp_sched, events["event_id"].to_list())):
        if t <= min_texp:
            texp_sched[i] = min_texp
        else:
            continue
    texp_sched = texp_sched / 1000
    events_texp = pd.DataFrame(
        {"event_id": events["event_id"].tolist(), "t_exp (ks)": texp_sched}
    )

    events_texp.to_csv(
        os.path.join(out_dir, "allsky_sched_full.txt"), index=False, sep=" "
    )

    ## batch
    batch_dir = os.path.join(out_dir, "texp_sched")
    os.makedirs(batch_dir, exist_ok=True)  ## weizmann
    list_of_lists = np.array_split(events_texp, N_out)
    batchnums = range(len(list_of_lists))
    for lst, num in zip(list_of_lists, batchnums):
        filename = os.path.join(batch_dir, "allsky_sched_batch" + str(num) + ".txt")
        lst.to_csv(filename, index=False, sep=",")

    return


if __name__ == "__main__":

    ## set up argparser
    parser = argparse.ArgumentParser(
        description="Re-aggregate the output from max-texp-by-sky-loc.py and prepare it for passing to the UVEX scheduler."
    )
    parser.add_argument("params", type=str, help="/path/to/params_file.ini")

    args = parser.parse_args()

    ## set up configparser
    config = configparser.ConfigParser()
    config.read(args.params)

    ## get info from params file
    out_dir = config.get("params", "save_directory")
    N_in = int(config.get("params", "N_batch_preproc", fallback=1))
    N_out = int(config.get("params", "N_batch_sched", fallback=N_in))
    band = config.get("params", "band")
    min_texp = float(config.get("params", "min_texp"))

    texp_dir = out_dir + "/texp_out/"

    os.makedirs(texp_dir, exist_ok=True)  ## weizmann

    ## run the script
    texp_cut_and_batch(texp_dir, out_dir, N_in, N_out, band, min_texp)
