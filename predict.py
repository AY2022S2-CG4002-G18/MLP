import pandas as pd
import numpy as np

from driver.driver import predictor


def convert_to_float(x):
    try:
        return float(x)
    except:
        return float(0)

def main():
    column_names = ['x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv("./driver/driver_content/sample_data.txt",
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')

    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    xs = df['x-axis'].values[:]
    ys = df['y-axis'].values[:]
    zs = df['z-axis'].values[:]
    segments.append([xs, ys, zs])

    # transform input
    # input is 240 flat
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 80, 3)
    reshaped_segments = reshaped_segments.reshape(reshaped_segments.shape[0], 240)

    # call predictor function
    print("Input Shape: " + str(reshaped_segments.shape))

    # print out prediction
    predictor(reshaped_segments)

main()