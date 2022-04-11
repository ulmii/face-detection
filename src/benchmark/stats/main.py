#! /usr/bin/env python3

import pandas as pd
import numpy as np

def calc_precision_recall(df: pd.DataFrame, ground_truth_count):
    if 'TP_FP' not in df:
        raise ValueError("TP_FP must be present in the Dataframe")

    work_df = df.copy()
    work_df['Precision'] = len(work_df) * [0.0]
    work_df['Recall'] = len(work_df) * [0.0]

    accum_recall = 0.0
    accum_precision = 0.0
    accum_rows = 1.0

    for index, row in work_df.iterrows():
        predicted = 1.0 if work_df.at[index, 'TP_FP'] == 'TP' else 0.0
        
        accum_recall += predicted
        accum_precision += predicted
        
        work_df.at[index, 'Precision'] = accum_precision / accum_rows
        work_df.at[index, 'Recall'] = accum_precision / ground_truth_count
        
        accum_rows += 1

    return work_df

def calc_ap(df: pd.DataFrame):
    if 'Precision' not in df:
        raise ValueError("Precision must be present in the Dataframe")
        
    if 'Recall' not in df:
        raise ValueError("Recall must be present in the Dataframe")

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], df['Recall'], [1.]))
    mpre = np.concatenate(([0.], df['Precision'], [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap