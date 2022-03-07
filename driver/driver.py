import pynq.lib.dma
import numpy as np
from pynq import allocate
from pynq import Overlay
import time

print("Running PYNQ Driver")
test_data = np.loadtxt('./driver_content/test_data/test_data.txt')
test_label_one_hot = np.loadtxt('./driver_content/test_data/test_label_one_hot.txt')
test_label_one_hot = np.loadtxt('./driver_content/test_data/test_label.txt')
print("Test data loaded")

BIT_PATH = "./driver_content/MLP2/mlp.bit"

class Driver:
    def __init__(self):
        self.ol = Overlay(BIT_PATH)
        self.dma = self.ol.axi_dma_0
        self.input_buffer = allocate(shape=(240,), dtype=np.int32)
        self.output_buffer = allocate(shape=(6,), dtype=np.int32)
    
    def predict(self, x):
        #quantise input
        x = (x * 1024).astype(np.int32)
        print("Initialise input buffer")
        self.input_buffer[:] = x
        print("Send data to DMA")
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.output_buffer)
        print("Waiting...")
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        return np.argmax(self.output_buffer, axis=0)
    
    def benchMark(self, x):
        print("Bench marking")


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
            dma.sendchannel.transfer(input_buffer)
            dma.recvchannel.transfer(output_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            return time.time() - start

driver = Driver()
result = driver.predict(test_data[0])
print(result)
