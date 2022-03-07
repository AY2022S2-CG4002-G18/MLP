from scipy import stats
import numpy as np
import tensorflow as tf

QUANT_FACTOR = 1024
QUANT = False

def create_segments_and_labels(df, time_steps, step, label_name):
    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def save_2d_weight_to_txt_file(np_array, filepath, quant=QUANT):
    """
    Takes in an np_array and filepath, convert the NP array into string and
    writes to the file
    """
    weights_to_write = []
    transposed = np_array.transpose()
    print(transposed.shape)
    for i in range(len(transposed)):
        weights_to_write_string = ""
        for j in range(len(transposed[i])):
            if quant:
                quant_val = transposed[i][j] * QUANT_FACTOR
                quant_val_int = int(quant_val)
                weights_to_write_string += str(quant_val_int)
            else:
                weights_to_write_string += str(transposed[i][j])
            weights_to_write_string += ","
        # replace trailing comma with }
        weights_to_write_string = weights_to_write_string[:-1]
        weights_to_write_string = "{" + weights_to_write_string + "}"

        # append to the collection of weights
        weights_to_write.append(weights_to_write_string)

    final_string = "{"
    for weight_string in weights_to_write:
        final_string = final_string + weight_string + ",\n"

    # remove trailing ','
    final_string = final_string[:-1]
    final_string = final_string + "}"

    with open(filepath, 'w') as f:
        f.write(final_string)


def save_2d_weight_to_dat_file(np_array, filepath, quant=QUANT):
    """
    Takes in an np_array and filepath, convert the NP array into string and
    writes to the file
    """
    weights_to_write = []
    transposed = np_array.transpose()
    for i in range(len(transposed)):
        weights_to_write_string = ""
        for j in range(len(transposed[i])):
            if quant:
                # try quantization
                quant_val = transposed[i][j] * QUANT_FACTOR
                quant_val_int = int(quant_val)
                weights_to_write_string += str(quant_val_int)
            else:
                weights_to_write_string += str(transposed[i][j])

            weights_to_write_string += ","
        # replace trailing comma with }
        weights_to_write_string = weights_to_write_string[:-1]

        # append to the collection of weights
        weights_to_write.append(weights_to_write_string)
        
    with open(filepath, 'w') as f:
        for weight_string in weights_to_write:
            weight_string = weight_string + "\n" 
            f.write(weight_string)


def save_1d_weight_to_txt_file(np_array, filepath, quant=QUANT):
    """
    Takes in an np_array and filepath, convert the NP array into string and
    writes to the file
    """
    weights_to_write_string = ""
    for i in range(len(np_array)):
        if quant:
            quant_val = np_array[i] * QUANT_FACTOR
            quant_val_int = int(quant_val)
            weights_to_write_string += str(quant_val_int)
        else:
            weights_to_write_string += str(np_array[i])
        
        weights_to_write_string += ","
            
    # replace trailing comma with }
    weights_to_write_string = weights_to_write_string[:-1]
    final_string = "{" + weights_to_write_string + "}"

    with open(filepath, 'w') as f:
        f.write(final_string)


def save_1d_weight_to_dat_file(np_array, filepath, quant=QUANT):
    """
    Takes in an np_array and filepath, convert the NP array into string and
    writes to the file
    """        
    with open(filepath, 'w') as f:
        for i in range(len(np_array)):
            weight_string = ""
            if quant:
                quant_val = np_array[i] * QUANT_FACTOR
                quant_val_int = int(quant_val)
                weight_string = str(quant_val_int)
            else:
                weight_string = str(np_array[i]) + "\n"
            f.write(weight_string)


def getMask(x):
    boolMask = tf.not_equal(x, 0)
    floatMask = tf.cast(boolMask, tf.float32)
    return floatMask