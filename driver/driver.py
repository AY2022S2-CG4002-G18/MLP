import pynq.lib.dma
from pynq import Overlay, allocate

import numpy as np
import time

print("Running PYNQ Driver")
test_data = np.loadtxt('./driver_content/test_data/test_data.txt', dtype=np.float32)
test_label_one_hot = np.loadtxt('./driver_content/test_data/test_label_one_hot.txt')
test_label = np.loadtxt('./driver_content/test_data/test_label.txt', dtype=np.int8)
print("Test data loaded")

BIT_PATH = "./driver_content/MLP2/mlp.bit"

class Driver:
    def __init__(self):
        print("Initiating dirver")
        self.ol = Overlay(BIT_PATH)
        self.hls_ip = self.ol.Hls_accel_0
        self.dma = self.ol.axi_dma_0

        self.hls_ip.write(0x00,0x01)
        
        self.input_buffer = allocate(shape=(240,), dtype=np.float32)
        self.output_buffer = allocate(shape=(6,), dtype=np.float32)
    
    def predict(self, x):
        #quantise input
        # x = (x * 1024).astype(np.int32)
        print("Initialise input buffer")
        for i in range(len(x)):
            self.input_buffer[i] = x[i]
        
        print("Send data to DMA")
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.output_buffer)
        print("Waiting to send...")
        self.dma.sendchannel.wait()
        print("Waiting to receive...")
        self.dma.recvchannel.wait()
        print("Return result")
        return self.output_buffer
        # return np.argmax(self.output_buffer, axis=0)
    
    def benchMark(self, x):
        print("Bench marking")
    
    def test(self, data, data_label):
        label_list = list(data_label)
        correct = 0
        total = len(data_label)
        for i in range(0,len(data)):
            result = self.predict(data[i])
            if result == label_list[i]:
                correct += 1
        return correct,total



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


def benchMarkAccuracy():
    label_list = list(test_label)
    correct = 0
    total = len(test_label)
    for i in range(0, len(test_data)):
        # create a new driver
        driver = Driver()
        buffer = driver.predict(test_data[i])
        result = np.argmax(buffer, axis=0)
        if (result == label_list[i]):
            print("Correct prediction", buffer, result, label_list[i])
            correct += 1
        else:
            print("Incorrect prediction", buffer, result, label_list[i])
        time.sleep(1.5)
    
benchMarkAccuracy()

# [-1046860800 -1029567616 -1030513536  1106247680  1121402496 -1023837696]
# [-1130766261 -1113345691 -1114172450  1022469131  1037687952 -1107492374]