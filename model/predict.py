from tensorflow import keras
import numpy as np

model = keras.models.load_model('04121653')

def predict(data):
    if len(data) < 192:
        # pad
        to_pad = 192 - len(data)
        for i in range(0,to_pad):
            data.append(0)
    np_input = np.array([data])
    float_arr = np_input.astype(np.float32)
    float_arr /= np.max(np.abs(float_arr), axis=0)

    y = model.predict(x=float_arr)
    
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
temp = [11076, 24544, 20954, 3230, -21201, -11926, -3582, -3984, -1011, 384, 4027, 9206, 8726, 11351, 20829, 23423, 22119, 14516, 3666, 426, -1704, -4587, -3910, -1512, 981, -582, -2826, -1920, -920, -1509, -4223, -3984, -1712, -1680, -1680, -1680, -1728, -1696, -1696, -1696, -1728, -1712, -1712, -1712, -1712, -1728, -1728, -1728, -1728, -1712, -1712, -1680, -1712, -1728, -1728, -1696, -1744, -1728, -1712, -1728, -1712, -1680, -1712, -1680, -1749, -497, -19, 28101, -11208, -3589, 1796, -8417, -8019, -1550, -6189, -3454, 7169, 3578, 13869, 6238, 6429, 2728, 2128, -1007, -5292, -1806, 1572, 489, 158, 1599, 1265, -3490, -4713, 1635, 3723, -1822, 31817, 31817, 31817, 6626, 19422, 16410, 114, -1366, 13274, 11370, 6766, 2138, -5706, -11070, -29122, -30346, -27422, -29230, -22950, -13286, -9338, -5490, -3406, -4466, -4686, -4850, -3890, -4126, -3862, -4730, -4266, -2758, -6048, 18980, 32467, 32467, 19648, 156, 3784, 13328, 24088, 26252, 21920, 11044, 11804, 9752, 5344, 12724, 14360, 15820, 16676, 21868, 20972, 16272, 13436, 11436, 10332, 13380, 15652, 17068, 16160, 14676, 16424, 17296, 5528, 480, -14304, -21552, 6508, 11152, 14148, 6768, 2024, 1172, -848, -488, -1828, -4080, -4020, -9140, -12844, -4160, -2328, -3088, -308, -608, -1052, -312, -524, -1108, -3152, -2748, -2180, -780, -996, -1812]
res = predict(temp[:192])
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
        
        if len(res) < 192:
            to_pad = 192 - len(res)
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