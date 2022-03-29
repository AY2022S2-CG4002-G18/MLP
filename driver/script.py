from pynq import Overlay, allocate
import pandas as pd
import numpy as np

print("Loading")
ol = Overlay("./mlp.bit")
dma = ol.axi_dma_0

input_buffer = allocate(shape=(276,), dtype=np.float32)
output_buffer = allocate(shape=(4,), dtype=np.float32)

data = pd.read_csv("./data_test.csv", header=None)
print("Loaded, start testing")

for i in range(len(data)):
    print(f"Testing {i}")
    np_input = data.iloc[[i]].to_numpy().astype(np.float32)
    # np_input = numpy.array(array).astype(np.float32)
    input_buffer[:] = np_input
    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    
    max_val = 0
    action_index = 0
    for i in range(0,len(output_buffer)):
        if output_buffer[i] > max_val:
            max_val = output_buffer[i]
            action_index = i
    print(output_buffer, action_index)
