import numpy as np

test_data = np.loadtxt('./driver_content/test_data/test_data.txt', dtype=np.float32)
test_label = np.loadtxt('./driver_content/test_data/test_label.txt', dtype=np.int8)

with open("./driver_content/test_data/test_data.dat", 'a') as f:
    for i in range(len(test_data)):
        data_list = list(test_data[i])
        write_str = ""
        for val in data_list:
            write_str += str(val)
            write_str += " "
        write_str += "\n"
        f.write(write_str)


with open("./driver_content/test_data/test_label.dat", 'a') as f:
    for i in range(len(test_label)):
        write_str = str(test_label[i]) + "\n"
        f.write(write_str)