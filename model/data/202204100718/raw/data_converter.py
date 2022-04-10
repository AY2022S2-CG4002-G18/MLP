import pandas as pd
import random
import numpy as np

FILENAME = "sw3.txt"
MAX_ID = 70 # The last data item
TEST_ID_RANGE = [60,70]
ROW_LENGTH = 32 * 6

class DataItem:
    def __init__(self):
        self.gx = []
        self.gy = []
        self.gz = []
        self.ax = []
        self.ay = []
        self.az = []
        self.arr = []
    
    def append_to_dataitem(self, data_array):
        self.gx.append(data_array[0])
        self.gy.append(data_array[1])
        self.gz.append(data_array[2])
        self.ax.append(data_array[3])
        self.ay.append(data_array[4])
        self.az.append(data_array[5])

        self.arr.append(data_array[0])
        self.arr.append(data_array[1])
        self.arr.append(data_array[2])
        self.arr.append(data_array[3])
        self.arr.append(data_array[4])
        self.arr.append(data_array[5])
    
    def convert_to_single_array(self):
        res = []
        res = self.gx + self.gy + self.gz + self.ax + self.ay + self.az
        
        if len(res) < ROW_LENGTH:
            to_pad = ROW_LENGTH - len(res)
            for i in range(0,to_pad):
                res.append(0)

        # normalize here
        arr = np.array(res)
        float_arr = arr.astype(np.float32)
        float_arr /= np.max(np.abs(float_arr), axis=0)
        normalized_list = float_arr.tolist()

        return normalized_list

class DataConverter:
    def __init__(self, f):
        self.file_to_convert = f
        return

    def convert_to_datafile(self):
        # data file name
        output_datafile = input("Enter the data file id\n")
        output_labelfile = output_datafile + "_label.txt"
        output_datafile += ".txt"

        # label
        label = input("Enter the label [0:reload, 1:shield, 2:grenade, 3:end]\n")

        # read CSV as dataframe
        df = pd.read_csv(self.file_to_convert, header=None)

        list_to_save = []

        yaw = df.iloc[:,0]
        pitch = df.iloc[:,1]
        roll = df.iloc[:,2]

        accx = df.iloc[:,3]
        accy = df.iloc[:,4]
        accz = df.iloc[:,5]

        list_to_save = [val for val in yaw]
        list_to_save = list_to_save + [val for val in pitch]
        list_to_save = list_to_save + [val for val in roll]
        list_to_save = list_to_save + [val for val in accx]
        list_to_save = list_to_save + [val for val in accy]
        list_to_save = list_to_save + [val for val in accz]
        list_to_save_string = str(list_to_save)[1:-1]

        if not len(list_to_save) == ROW_LENGTH:
            print("Corrupted row")

        f = open(output_datafile, "a")
        f.write(list_to_save_string)

        f = open(output_labelfile, "a")
        f.write(str(label))
        
        return
    
    def convert_input(self):
        # Not used
        return

    def get_as_single_array(self):
        # Not used
        return

    def merge_data(self, ID_START, ID_END, output_file_name):
        data_matrix = []

        for index in range(ID_START+1,ID_END+1):
            filename = str(index) + ".txt"
            df = pd.read_csv(filename, header=None)
            data_arr = df.to_numpy()[0].tolist()
            
            if len(data_arr) < ROW_LENGTH:
                # pad until ROW_LENGTH
                to_pad = ROW_LENGTH - len(data_arr)
                for i in range(to_pad):
                    data_arr.append(0)
            elif len(data_arr) > ROW_LENGTH:
                # reduce until ROW_LENGTH
                data_arr = data_arr[:ROW_LENGTH]
                
            data_matrix.append(data_arr)
        
        # write to a single csv
        with open(output_file_name, "a") as f:
            for arr in data_matrix:
                array_to_write = str(arr)
                array_to_write = array_to_write[1:-1] + "\n"
                f.write(array_to_write)
        return
    
    def merge_label(self, ID_START, ID_END, output_file_name):
        labels = []
        for index in range(ID_START+1, ID_END+1):
            filename = str(index) + "_label.txt"
            df = pd.read_csv(filename, header=None)
            label = df.to_numpy().tolist()[0][0]
            labels.append(label)

        # write to a single csv
        with open(output_file_name, "a") as f:
            for label in labels:
                label_to_write = str(label) + "\n"
                f.write(label_to_write)
        return

    def create_aggregate(self, file1, file2, label):
        output_datafile = input("Enter the data file id\n")
        output_labelfile = output_datafile + "_label.txt"
        output_datafile += ".txt"

        # read CSV as dataframe
        df_1 = pd.read_csv(file1, header=None)
        l1 = df_1.to_numpy()[0].tolist()
        df_2 = pd.read_csv(file2, header=None)
        l2 = df_2.to_numpy()[0].tolist()

        # get list 1
        if len(l1) < ROW_LENGTH:
                # pad until ROW_LENGTH
            to_pad = ROW_LENGTH - len(l1)
            for i in range(to_pad):
                l1.append(0)
        elif len(l1) > ROW_LENGTH:
            # reduce until ROW_LENGTH
            l1 = l1[:ROW_LENGTH]
        
        # get list 2
        if len(l2) < ROW_LENGTH:
                # pad until ROW_LENGTH
            to_pad = ROW_LENGTH - len(l2)
            for i in range(to_pad):
                l2.append(0)
        elif len(l1) > ROW_LENGTH:
            # reduce until ROW_LENGTH
            l2 = l2[:ROW_LENGTH]

        # generate list 3
        i = 0
        l3 = []
        while i < ROW_LENGTH:
            avg = int((l1[i] + l2[i])/2)
            l3.append(avg)
            i += 1
        
        l3str = str(l3)[1:-1]
        f = open(output_datafile, "a")
        f.write(l3str)

        f = open(output_labelfile, "a")
        f.write(str(label))
        return

    def append_to_datafile_from_single_source(self, source_file, label, target_data_file, target_label_file):
        data = []
        labels = []
        
        with open(source_file, 'r') as f:
            lines = f.readlines()
            data_item = DataItem()    
            for line in lines:
                if line == str(label) + '\n' or line == str(label):
                    # end of current data item
                    res = data_item.convert_to_single_array() # convert into single array
                    if len(res) == ROW_LENGTH:
                        data.append(res)
                        labels.append(label)
                    else:
                        print("Skip inconsistent length row")

                    # reset data struct
                    data_item = DataItem()
                    continue

                str_tokens = line[:-1].split(',') # remove \n

                try:
                    tokens = [int(val) for val in str_tokens] # convert to int
                except ValueError:
                    print(tokens, str_tokens, line)

                # initialise data item
                if len(tokens) == 6:
                    data_item.append_to_dataitem(tokens)
        
        with open(target_data_file, 'a') as f:
            for row in data:
                row_str = str(row)[1:-1] + '\n'
                f.write(row_str)

        with open(target_label_file, 'a') as f:
            for label in labels:
                label_str = str(label) + '\n'
                f.write(label_str)
        return

    def train_test_split(self, data_file, label_file, x_train, y_train, x_test, y_test, test_percentage):
        data = []
        labels = []
        with open(data_file, 'r') as dataf:
            data = dataf.readlines()
        with open(label_file, 'r') as labelf:
            labels = labelf.readlines()
        
        if not len(data) == len(labels):
            print("Mismatch in data label dimensions")
            return
        
        x_train = open(x_train, 'a')
        y_train = open(y_train, 'a')
        x_test = open(x_test, 'a')
        y_test = open(y_test, 'a')

        for i in range(0,len(data)):
            if random.uniform(0, 1) < test_percentage:
                x_test.write(data[i])
                y_test.write(labels[i])
            else:
                x_train.write(data[i])
                y_train.write(labels[i])

    def verify_datafile(self, data_file):
        with open(data_file) as f:
            lines = f.readlines()
            for line in lines:
                str_tokens = line[:-1].split(',')
                if not len(str_tokens) == ROW_LENGTH:
                    print(line)
                    return 
        
dc = DataConverter(FILENAME)
# dc.convert_to_datafile()


# dc.append_to_datafile_from_single_source("data1.txt", 1, "data.csv", "label.csv")
# dc.append_to_datafile_from_single_source("data2.txt", 2, "data.csv", "label.csv")
# dc.append_to_datafile_from_single_source("data3.txt", 3, "data.csv", "label.csv")

# 202204100718
# dc.append_to_datafile_from_single_source("data0jo.txt", 0, "data0.csv", "label0.csv")
# dc.append_to_datafile_from_single_source("data0lp.txt", 0, "data0.csv", "label0.csv") # Has one error row, corrected
# dc.append_to_datafile_from_single_source("data0sug.txt", 0, "data0.csv", "label0.csv") # Has three error rows, corrected
# dc.append_to_datafile_from_single_source("data0bobnhaitao.txt", 0, "data0.csv", "label0.csv") # Has three error rows, corrected

# dc.append_to_datafile_from_single_source("data1jo.txt", 1, "data1.csv", "label1.csv")
# dc.append_to_datafile_from_single_source("data1ht.txt", 1, "data1.csv", "label1.csv")
# dc.append_to_datafile_from_single_source("data1lp.txt", 1, "data1.csv", "label1.csv")
# dc.append_to_datafile_from_single_source("data1bob.txt", 1, "data1.csv", "label1.csv")
# dc.append_to_datafile_from_single_source("data1sug.txt", 1, "data1.csv", "label1.csv")

# dc.append_to_datafile_from_single_source("data2jo.txt", 2, "data2.csv", "label2.csv")
# dc.append_to_datafile_from_single_source("data2lp.txt", 2, "data2.csv", "label2.csv")
# dc.append_to_datafile_from_single_source("data2bobnht.txt", 2, "data2.csv", "label2.csv")
# dc.append_to_datafile_from_single_source("data2sug.txt", 2, "data2.csv", "label2.csv")

# dc.append_to_datafile_from_single_source("data3jo.txt", 3, "data3.csv", "label3.csv")
# dc.append_to_datafile_from_single_source("data3lp.txt", 3, "data3.csv", "label3.csv")
# dc.append_to_datafile_from_single_source("data3bobnhaitao.txt", 3, "data3.csv", "label3.csv")
# dc.append_to_datafile_from_single_source("data3sug.txt", 3, "data3.csv", "label3.csv")
# ---- done ----

dc.append_to_datafile_from_single_source("data0jo.txt", 0, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data0lp.txt", 0, "data.csv", "label.csv") # Has one error row, corrected
dc.append_to_datafile_from_single_source("data0sug.txt", 0, "data.csv", "label.csv") # Has three error rows, corrected
dc.append_to_datafile_from_single_source("data0bobnhaitao.txt", 0, "data.csv", "label.csv") # Has three error rows, corrected

dc.append_to_datafile_from_single_source("data1jo.txt", 1, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data1ht.txt", 1, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data1lp.txt", 1, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data1bob.txt", 1, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data1sug.txt", 1, "data.csv", "label.csv")

dc.append_to_datafile_from_single_source("data2jo.txt", 2, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data2lp.txt", 2, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data2bobnht.txt", 2, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data2sug.txt", 2, "data.csv", "label.csv")

dc.append_to_datafile_from_single_source("data3jo.txt", 3, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data3lp.txt", 3, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data3bobnhaitao.txt", 3, "data.csv", "label.csv")
dc.append_to_datafile_from_single_source("data3sug.txt", 3, "data.csv", "label.csv")
# dc.verify_datafile('data.csv')
# dc.verify_datafile('data_test.csv')

dc.train_test_split('data.csv', 'label.csv', 'data_train.csv', 'label_train.csv', 'data_test.csv', 'label_test.csv', 0.10)
# merge to test
# dc.merge_data(TEST_ID_RANGE[0], TEST_ID_RANGE[1], "data_test.csv")
# merge to train
# dc.merge_data(0, MAX_ID, "data.csv")
# merge to test
# dc.merge_label(TEST_ID_RANGE[0], TEST_ID_RANGE[1], "label_test.csv")
# merge to train
# dc.merge_label(0, MAX_ID, "label.csv")

# FILE1 = "31.txt"
# FILE2 = "33.txt"
# LABEL = 3
# dc.create_aggregate(file1=FILE1, file2=FILE2, label=LABEL)

# df = pd.read_csv("3.txt", header=None)
# data_arr = df.to_numpy()[0].tolist()
# print(len(data_arr))
# print(str(data_arr)[1:-1])