from pynq import Overlay, allocate
import numpy as np

ol = Overlay("./mlp0239.bit")
dma = ol.axi_dma_0

input_buffer = allocate(shape=(276,), dtype=np.float32)
output_buffer = allocate(shape=(4,), dtype=np.float32)

data = pd.read_csv("./data_test.csv", header=None)

for i in range(len(data)):
    np_input = data.iloc[[0]].to_numpy().astype(np.float32)
    input_buffer[:] = np_input
    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    output_buffer
