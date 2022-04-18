#! /usr/bin/env python3

import pandas as pd
import numpy as np

def calc_precision_recall(df: pd.DataFrame, ground_truth_count):
    if 'TP_FP@25' not in df:
        raise ValueError("TP_FP@25 must be present in the Dataframe")
    if 'TP_FP@50' not in df:
        raise ValueError("TP_FP@50 must be present in the Dataframe")
    if 'TP_FP@75' not in df:
        raise ValueError("TP_FP@75 must be present in the Dataframe")
    
    work_df = df.copy()
    work_df['Precision@25'] = len(work_df) * [0.0]
    work_df['Recall@25'] = len(work_df) * [0.0]
    work_df['Precision@50'] = len(work_df) * [0.0]
    work_df['Recall@50'] = len(work_df) * [0.0]
    work_df['Precision@75'] = len(work_df) * [0.0]
    work_df['Recall@75'] = len(work_df) * [0.0]

    accum_precision_25 = 0.0
    accum_precision_50 = 0.0
    accum_precision_75 = 0.0
    accum_rows = 1.0

    for index, row in work_df.iterrows():
        predicted_25 = 1.0 if work_df.at[index, 'TP_FP@25'] else 0.0
        predicted_50 = 1.0 if work_df.at[index, 'TP_FP@25'] else 0.0
        predicted_75 = 1.0 if work_df.at[index, 'TP_FP@25'] else 0.0

        accum_precision_25 += predicted_25
        accum_precision_50 += predicted_50
        accum_precision_75 += predicted_75
        
        work_df.at[index, 'Precision@25'] = accum_precision_25 / accum_rows
        work_df.at[index, 'Recall@25'] = accum_precision_25 / ground_truth_count

        work_df.at[index, 'Precision@50'] = accum_precision_50 / accum_rows
        work_df.at[index, 'Recall@50'] = accum_precision_50 / ground_truth_count

        work_df.at[index, 'Precision@75'] = accum_precision_75 / accum_rows
        work_df.at[index, 'Recall@75'] = accum_precision_75 / ground_truth_count
        
        accum_rows += 1

    return work_df

def calc_ap(precision, recall):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], precision, [1.]))
    mpre = np.concatenate(([0.], recall, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap