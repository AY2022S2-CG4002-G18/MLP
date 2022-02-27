from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.1f}'.format
plt.style.use('ggplot')

def read_data(file_path):
    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
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

    return df

def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.nan
 
def show_basic_dataframe_info(dataframe):
    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

def get_data_frame(file_path):
    # Load data set containing all the data from csv
    return read_data(file_path)

def visualize_data(df):
    # Use matplot to visualize data
    # Show how many training examples exist for each of the six activities
    df['activity'].value_counts().plot(kind='bar',
                                    title='Training Examples by Activity Type')
    plt.show()
    # Better understand how the recordings are spread across the different
    # users who participated in the study
    df['user-id'].value_counts().plot(kind='bar',
                                    title='Training Examples by User')
    plt.show()

    for activity in np.unique(df['activity']):
        subset = df[df['activity'] == activity][:180]
        plot_activity(activity, subset)
    return

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def read_and_transform_input(filepath):
    # x, y, z acceleration as features
    return