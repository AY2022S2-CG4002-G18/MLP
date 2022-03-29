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

BIT_PATH = "./driver_content/MLP2/mlp.bit"

test_input_90 = test_data[0]
test_input_90 = test_input_90[0:90]


class Driver:
    def __init__(self):
        self._initialise()

    def _initialise(self):
        print("Initiating dirver")
        self.ol = Overlay(BIT_PATH)
        self.ol.reset()
        self.dma = self.ol.axi_dma_0

        print("Allocated buffer")
        self.input_buffer = allocate(shape=(240,), dtype=np.int32)
        self.output_buffer = allocate(shape=(6,), dtype=np.int32)
    
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
        
        print("Waiting on send channel")
        self.dma.sendchannel.wait()

        print("Transfer output buffer")
        self.dma.recvchannel.transfer(self.output_buffer)
        
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
    input_buffer = allocate(shape=(240,), dtype=np.float32)
    output_buffer = allocate(shape=(6,), dtype=np.float32)
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
benchMark()


