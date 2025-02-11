import pandas as pd
import cv2
from math import copysign, log10
import numpy as np

metadata_dir = "pf-bula-castillo-garcia/annotationsv1.csv"
annotations = pd.read_csv(metadata_dir)
len_df = annotations.shape[0]

for j in range(7):
    annotations[f'HuMoment_{j}'] = np.nan 

for i in range(len_df):
    row = annotations.iloc[i]
    im = cv2.imread(f'{row.directory_path}', cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_TRUNC) # Trunc porque la radiografia NO son solo #00000 y #FFFFF
    
    #  Moments 
    moments = cv2.moments(im) 
    
    #  Hu Moments 
    huMoments = cv2.HuMoments(moments)

   
    for j in range(7):
        if huMoments[j, 0] != 0:
            huMoments[j, 0] = -1 * copysign(1.0, huMoments[j, 0]) * log10(abs(huMoments[j, 0]))
        else:
            huMoments[j, 0] = 0 
    
    
    for j in range(7):
        annotations.at[i, f'HuMoment_{j}'] = huMoments[j, 0]

annotations.to_csv("pf-bula-castillo-garcia/annotationsv2.csv")