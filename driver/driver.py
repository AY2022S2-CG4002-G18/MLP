import pynq.lib.dma
from pynq import Overlay, allocate

import numpy as np
from time import time
import time as tt

# export XILINX_XRT=/usr
print("Running PYNQ Driver")
test_data = np.loadtxt('./driver_content/test_data/test_data.txt', dtype=np.float32)
test_label_one_hot = np.loadtxt('./driver_content/test_data/test_label_one_hot.txt')
test_label = np.loadtxt('./driver_content/test_data/test_label.txt', dtype=np.int8)
print("Test data loaded")

BIT_PATH = "./driver_content/MLP5/mlp.bit"

test_input_90 = test_data[0]
test_input_90 = test_input_90[0:90]

# ol.is_loaded
# ol.bitfile_name
# ol.pr_dict
class Driver:
    def __init__(self):
        self._initialise()

    def _initialise(self):
        print("Initiating driver")
        self.ol = Overlay(BIT_PATH)
        # self.ol.reset()
        self.dma = self.ol.axi_dma_0

        print("Allocated buffer")
        self.input_buffer = allocate(shape=(276,), dtype=np.int32)
        self.output_buffer = allocate(shape=(4,), dtype=np.int32)
    
    def predict_non_verbose(self, x):
        x = x.astype(np.float32)
        for i in range(len(x)):
            self.input_buffer[i] = x[i]
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.output_buffer)
        self.dma.sendchannel.wait()
        # self.dma.recvchannel.wait()
        return self.output_buffer

    def predict(self, x):
        #quantise input
        # x = (x * 1024).astype(np.int32)
        x = x.astype(np.int32)
        print("Initialise input buffer")
        for i in range(len(x)):
            self.input_buffer[i] = x[i]
        
        print("Send data to DMA")
        self.dma.sendchannel.transfer(self.input_buffer)
        print("Transfer output buffer")
        self.dma.recvchannel.transfer(self.output_buffer)
        
        print("Waiting on send channel")
        self.dma.sendchannel.wait()
        print("Waiting to receive...")
        self.dma.recvchannel.wait()
        
        print("Return result")
        return self.output_buffer
        # return np.argmax(self.output_buffer, axis=0)
    
    def test(self, data, data_label):
        label_list = list(data_label)
        correct = 0
        total = len(data_label)
        for i in range(0,len(data)):
            result = self.predict(data[i])
            if result == label_list[i]:
                correct += 1
        return correct,total

def predict_once():
    ol = Overlay(BIT_PATH)
    x = test_data[0].astype(np.float32)
    
    dma = ol.axi_dma_0
    input_buffer = allocate(shape=(276,), dtype=np.float32)
    output_buffer = allocate(shape=(4,), dtype=np.float32)
    input_buffer[:] = x
    # print(input_buffer)
    print("Sent buffer")

    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(output_buffer)

    dma.sendchannel.wait()
    # dma.recvchannel.wait()

    return output_buffer

def measure_time(x):
    # quantise input
    x = (x * 255).astype(np.int32)
    # load overlay
    ol = Overlay("./driver_content/design_2.bit")
    dma = ol.axi_dma_0
    with allocate(shape=(x.shape), dtype=np.int32) as input_buffer:
        input_buffer[:] = x
        with allocate(shape=(16,), dtype=np.int32) as output_buffer:
            start = time.time()
            dma.sendchannel.transfer(input_buffer) # Transfer the input_buffer to send DMA
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait() # Ensures the dMA transactions have completed
            return time.time() - start

def benchMark():
    label_list = list(test_label)
    correct = 0
    total = len(test_label)
    total_time_used = []

    for i in range(0, len(test_data)):
        # create a new driver
        total += 1
        driver = Driver()
        time_start = time()
        buffer = driver.predict(test_data[i])
        time_used = time() - time_start
        total_time_used.append(time_used)
        result = np.argmax(buffer, axis=0)
        if (result == label_list[i]):
            print("Correct prediction", buffer, result, label_list[i])
            correct += 1
        else:
            print("Incorrect prediction", buffer, result, label_list[i])
    
    print(correct/total)
    print("Time Used")
    print(time_used)

# run bench marking - 100 cases
# benchMark()

driver = Driver()

temp1 = [1086, 13656, -11826, -12720, -55, 915, -4042, -2729, -2610, 3321, 9982, 11970, -2926, -9147, 2389, -2375, 36, -2001, -3935, -1925, -2454, -2009, -3447, -8404, -11140, -14449, -20114, -9279, -383, 1861, -500, -2697, -1754, -958, -276, 247, 336, 191, -138, -233, 22, 362, 417, -69, -183, -99, -2160, -2144, -2176, -2176, -2160, -2176, -2160, -2176, -2144, -2144, -2176, -2160, -2160, -2160, -2144, -2176, -2176, -2144, -2160, -2128, -2192, -2144, -2176, -2176, -2160, -2160, -2176, -2144, -2176, -2160, -2128, -2176, -2176, -2176, -2160, -2160, -2144, -2144, -2176, -2160, -2160, -2176, -2176, -2144, -2192, -2160, -14064, 27340, -21250, 1273, 2023, 3037, -8809, 1063, -2936, 9200, 19926, -32599, 11367, -4500, -6941, -4285, -7326, -13334, -16993, -23345, -27057, -22028, -21431, -15834, -12606, -17167, -14056, -17766, 3120, 4217, -524, -6004, -3781, 180, -8, -695, -521, 281, -387, -599, 89, 425, 114, -416, -371, -354, 13874, 1586, 550, 5626, 9950, 4326, 5074, -5666, -5286, -7014, -12366, -20662, -19870, -12998, -11466, -10650, -9218, -9982, -9826, -8614, -7234, -6238, -5826, -3330, -14, 2822, 8558, 34, -1390, -942, -1058, -1318, -1570, -1278, -398, -782, -1066, -1310, -946, -970, -950, -1122, -1322, -1286, -1206, -1198, 21944, 16372, 14928, 14228, 19112, 15136, 9952, 8912, 10004, 8316, 3848, 10324, 23928, 25080, 21896, 19808, 17552, 17544, 14448, 13344, 13420, 9304, 9448, 7192, 9808, 5320, 8292, -1040, -832, 5300, 5472, 3984, 1780, 1900, 3252, 3184, 2356, 2780, 3180, 2856, 2852, 3084, 3500, 3140, 2972, 3064, 5164, 7576, 13864, 9176, 5584, 7664, 5556, 5540, 7836, 5072, 5296, 3888, 13228, 4784, 3516, 5976, 2252, 4440, 3292, 6000, 6992, 8516, 9724, 10596, 14180, 13728, 32767, 11716, 18744, 17136, 17776, 17736, 17972, 18092, 17592, 17880, 17980, 18080, 17876, 17856, 17820, 17748, 17848, 17948, 17808, 17724]
np_input = np.array([temp1])
buffer = driver.predict(np_input)
print(buffer)


temp2 = [-18937, -13363, 12805, -13606, -1080, 2697, -2935, -2969, -2854, -2175, 777, -134, -638, -3395, -4723, -19, 6215, 10899, 14441, 7882, -12028, -16303, -2006, 8285, 7392, 1713, -39, -2186, -3609, -4640, -5576, -5026, -1931, -4617, -10560, -8009, -6158, -18137, -20194, -22245, -14115, -4113, -2271, -572, 2979, 4242, -2144, -2144, -2144, -2176, -2192, -2192, -2160, -2160, -2160, -2144, -2144, -2144, -2192, -2160, -2160, -2160, -2176, -2144, -2144, -2176, -2160, -2160, -2160, -2160, -2176, -2144, -2176, -2192, -2160, -2160, -2160, -2176, -2112, -2176, -2144, -2144, -2144, -2160, -2160, -2160, -2192, -2176, -2176, -2144, -2144, -2176, -27768, 21067, -14332, -10999, 10046, 313, -2279, 4500, 5927, 3082, 3871, 6547, -1817, -9729, -2864, 8957, 13054, 10496, 15381, 22270, 21971, 3638, -9979, -11816, -6764, -10806, -10125, -9992, -9402, -11155, -20743, -30895, -30273, -26850, -31426, -32598, -27742, -29985, -29863, -11352, -5773, 661, 1169, 6729, 7473, 1515, 31817, 31817, 5694, -10178, 922, -1790, 1342, 2490, 3398, 5326, 9058, 11954, 13230, 12306, 5990, 170, -5066, -12926, -20382, 31818, -31394, -19106, -12778, -13850, -12922, -9098, -8342, -9414, -9774, -9790, -9994, -9170, -9546, -9394, -6818, -2946, -626, 1830, 5422, 11118, 5534, 4966, 1290, 686, -218, -1786, 32467, 32467, 16164, -4036, 4592, 13972, 10388, 9556, 13224, 10520, 12924, 15860, 16500, 12300, 4412, 9376, 12676, 14412, 18644, 11040, 22564, 21000, 22080, 14436, 15804, 14232, 11820, 11872, 12868, 14632, 16556, 13404, 9936, 11768, 10840, 4184, 7628, 7268, -1380, -3968, -2040, -748, -1932, -1384, 484, 4812, 3820, 7976, 1252, 12588, 11856, 9052, 10980, 12232, 8944, 6296, 2688, 3464, 968, 3860, 3308, 132, 592, 84, 6660, 14520, 20784, 12436, 6776, 6980, 8284, 5396, 4276, 6100, 6380, 6700, 8700, 7752, 7648, 12352, 9724, 6548, 16412, 21436, 15824, 32767, 9872, 17404, 18448, 17844, 17096, 18224]
np_input2 = np.array([temp2])
buffer2 = driver.predict(np_input2)
print(buffer2)