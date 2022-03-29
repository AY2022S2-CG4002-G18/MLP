from tensorflow import keras
import numpy as np

model = keras.models.load_model('03282139')

def predict(data):
    if len(data) < 240:
        # pad
        to_pad = 240 - len(data)
        for i in range(0,to_pad):
            data.append(0)
    np_input = np.array([data])
    y = model.predict(x=np_input)
    
    max_index = 0
    max_prob = 0
    for index in range(0,4):
        if y[0][index] > max_prob:
            max_prob = y[0][index]
            max_index = index

    return max_index

def predict_with_bit(data):
    return 1

# copy paste the array here
temp = [-32558, 31647, 12840, 3461, 19339, 23002, 25860, 25893, 26866, 27340, 29336, 28594, -32559, -9476, -26250, -9321, -16753, -21553, -26105, -20113, -32558, -32558, -32559, 10495, 28971, -32559, 30812, -32559, 19959, 14216, 17578, -26250, -2710, -4233, -10224, -15885, -16672, -20983, -27997, -1504, -1504, -1488, -1504, -1488, -1504, -1488, -1472, -1488, -1520, -1504, -1488, -1504, -1488, -1504, -1488, -1504, -1504, -1520, -1504, -1520, -1504, -1488, -1504, -1520, -1504, -1520, -1488, -1488, -1520, -1536, -1488, -1520, -1504, -1472, -1488, -1520, -1520, -1472, -22123, -32599, -32598, -2796, -1092, -640, 6139, 2412, 1941, -3825, -4080, -5453, -3748, -19444, 6707, -3078, 4660, 2320, -2566, 5986, 15442, -9584, -32599, -22155, -5757, 15388, 7279, 15128, 16954, -481, 894, -8217, 4431, 11954, 11187, 2438, 4869, 7553, 3623, 31817, 19042, -1198, 6870, 5234, 4250, -3206, -3866, -10406, -12010, -19306, -22814, -26310, 31818, -19802, -25034, -23626, -15758, -18238, -18366, 3910, 31817, 18546, 2262, 16554, 3050, -8414, -16002, -27970, -24998, -27466, -32394, -14270, -15206, -14570, -12878, -11122, -9350, -6386, -2088, 10460, -6612, -5280, -2488, -3488, -5384, -3024, -6056, 1300, 1416, 2280, -16848, -9632, -5340, -2840, -3056, -4304, -6816, -11236, -4764, 32467, -2728, 4208, 1812, -472, -2676, -1292, -6988, -7468, -27564, -23528, -752, -1160, -3552, -3976, -3760, -1756, -1628, 32228, 4952, 18260, 21852, 11716, 17072, 11680, 11920, 8496, 9420, 3884, -5428, -7340, -10136, -21800, -12444, -3752, 3016, -2264, -10604, 5580, 26196, 3048, 25116, 11500, 14508, 7148, 2836, 4584, -688, -23244, 11244, 8052, 9104, 9216, 9728, 8428, 12428, 11100, 0, 0, 0, 0, 0, 0]
res = predict(temp)
print("Predicted result:",res)

# Helper class
class DataItem:
    def __init__(self):
        self.gx = []
        self.gy = []
        self.gz = []
        self.ax = []
        self.ay = []
        self.az = []
    
    def append_to_dataitem(self, data_array):
        self.gx.append(data_array[0])
        self.gy.append(data_array[1])
        self.gz.append(data_array[2])
        self.ax.append(data_array[3])
        self.ay.append(data_array[4])
        self.az.append(data_array[5])
    
    def convert_to_single_array(self):
        res = []
        res = self.gx + self.gy + self.gz + self.ax + self.ay + self.az
        
        if len(res) < 240:
            to_pad = 240 - len(res)
            for i in range(0,to_pad):
                res.append(0)

        return res

# you can use the function below to print out the array
def print_as_one_d(filename):
    with open(filename, 'r') as f:
        data_item = DataItem()

        lines = f.readlines()

        for line in lines:
            str_tokens = line.replace("\n","").split(',') # remove \n
            tokens = [int(val) for val in str_tokens] # convert to int
        
            # initialise data item
            if len(tokens) == 6:
                data_item.append_to_dataitem(tokens)
    
    # Output array in terminal
    print(data_item.convert_to_single_array())

# print out test array, notice it's padded automatically by dataitem class
print_as_one_d("test")