from pynq import Overlay, Xlnk
import pynq.lib.dma
import numpy as np
import pandas as pd
import csv
from random import randint
import os
import time

ip_timesteps, ip_features = 50, 6
op_moves = 9 
c1_kernel_size, c1_features, c1_kernels, c1_bias = 3, 6, 64, 64
c2_kernel_size, c2_features, c2_kernels, c2_bias = 3, 64, 64, 64
d1_features, d1_units, d1_bias = 1472, 100, 100
d2_features, d2_units, d2_bias = 100, op_moves, op_moves 
QUANT_FACTOR = 128
moves = ["gun","hair","sidepump","dab","elbowkick","logout","listen","pointhigh","wipetable"]

weights_dir = "/home/xilinx/neural_net/params/quantized_wb"
overlay_path = "/home/xilinx/neural_net/ConvNN.bit"
input_file = os.path.join(os.getcwd(), "params", "inputs.dat")

ip_file = os.path.join(os.getcwd(), "data", "testX.txt")
sw_op_file = os.path.join(os.getcwd(), "final_op.dat")
act_op_file = os.path.join(os.getcwd(), "data", "testy.txt")
compare_file = os.path.join(os.getcwd(), "compare.txt")

# Files must be in the exact order of the input stream
wb_filenames = ["conv1d0_w_q_512.dat","conv1d0_b_q_512.dat","conv1d1_w_q_512.dat","conv1d1_b_q_512.dat","dense4_w_q_512.dat","dense4_b_q_512.dat","dense5_w_q_512.dat","dense5_b_q_512.dat"]



def main():
    driver = CnnDriver()
    evaluate(driver, compare_sw=False)

def evaluate(driver,compare_sw=True,save=False):
    '''
    Main evaluation function for hardware implemented CNN.
    
    compare_sw determines if the implemented CNN should be compared to
    the software implementation, or the actual correct dance move
    
    save determines if predicted and actual probabilities should be saved
    in a text file
    '''
    if compare_sw:
        inputs, corr_op, size = get_test_ip_op(ip_file, sw_op_file)
    else:
        inputs, corr_op, size = get_test_ip_op(ip_file, act_op_file, ip_delimiter=' ', op_delimiter=' ')
    
    #Evaluation metrics
    mismatch_count = 0
    duration_sum = 0
    
    #Save file in 
    if save:
        data = np.empty((size,2*op_moves))
    
    print("Mismatches:\n")
    for i in range(size):
        reshaped_ip = inputs[i].reshape(ip_timesteps,ip_features)
        reshaped_op = corr_op[i].reshape(op_moves,)

        move, predict_op, duration = driver._predict_once(reshaped_ip,verbose=True)
        correct_move = moves[np.argmax(reshaped_op)]
        
        #Evaluation metrics update
        duration_sum += round(duration,5)
        if (move != correct_move):
            mismatch_count += 1
            compare_outputs(predict_op, reshaped_op)
        
        #Record data in numpy array
        if save:
            entry = np.concatenate((predict_op,reshaped_op),axis=0)
            data[i] = entry

    if save:
        np.savetxt(compare_file,data)
    
    if compare_sw:
        print("\nThis model is " + str(round((size-mismatch_count)/size,3)*100) + "% accurate with respect to software implementation.")
    else: 
        print("\nThis model is " + str(round((size-mismatch_count)/size,3)*100) + "% accurate with respect to actual moves..")
    print("Average prediction duration: %s s" % round(duration_sum/size,4))    


def get_test_ip_op(input_file, output_file, ip_delimiter=' ', op_delimiter=','):

    ip = np.genfromtxt(input_file,delimiter=ip_delimiter)
    op = np.genfromtxt(output_file,delimiter=op_delimiter)
    ip_size = ip.shape[0]
    op_size = op.shape[0]
    assert(ip_size == op_size)
    all_inputs = np.split(ip,ip_size,axis=0)
    all_outputs = np.split(op,op_size,axis=0)
    
    return all_inputs, all_outputs, ip_size

def compare_outputs(predict_op, actual_op):
    """
    Compare predicted probabilities of dance moves with actual
    probabilities from software model
    """
    print("\nPredicted output:")
    for j in range(len(predict_op)):
        print(moves[j] + ": " + str(predict_op[j]))
    print("Predicted move: " + moves[np.argmax(predict_op)])
    print("Actual output:")
    for j in range(len(actual_op)):
        print(moves[j] + ": " + str(actual_op[j]))
    print("Actual move: " + moves[np.argmax(actual_op)])

def predict_single_window(window_file):
    """
    Given a file containing a single window of data, print the move predicted
    """
    dummy_input = np.genfromtxt(window_file,delimiter=',')
    move = driver._predict_once(dummy_input)
    print(move)



class CnnDriver:
    """
    Driver class to be used by comms manager thread.
    """

    def __init__(self):
        self._get_buffer_size()
        self.ip_buf_index = 0
        self._start_dma()
        self._load_weights()

    def _get_buffer_size(self):
        '''
        Determine internal input and output buffer sizes
        from number of CNN parameters
        '''
        c1_ip = c1_kernel_size * c1_features * c1_kernels + c1_bias
        c2_ip = c2_kernel_size * c2_features * c2_kernels + c2_bias
        d1_ip = d1_features * d1_units + d1_bias
        d2_ip = d2_features * d2_units + d2_bias
        act_ip = ip_timesteps * ip_features
        
        self.w_buf_size = c1_ip + c2_ip + d1_ip + d2_ip
        self.ip_buf_size = self.w_buf_size + act_ip
        self.op_buf_size = op_moves


    def _start_dma(self):
        '''
        Call the .bit, .hwh and .tcl files and write them to
        the PL hardware in the FPGA i.e. initialise the hardware
        for the CNN
        '''
        ol = Overlay(overlay_path)

        self.xlnk = Xlnk()
        self.dma = ol.axi_dma_0

    def _load_file(self, filepath):
        '''
        Load weights from a single file inito internal numpy
        array in the object
        '''
        inputs = np.genfromtxt(filepath,delimiter=',')
        inputs = np.round(inputs) #Remove if accuracy is poort
        for element in np.nditer(inputs):
            self.w_arr[self.ip_buf_index] = element
            self.ip_buf_index += 1
        print("Loaded file " + filepath + " successfully.")

    def _load_weights(self):
        '''
        Load weights from given file directory into
        internal numpy array in the object
        '''
        self.w_arr = np.empty(shape=(self.w_buf_size,),dtype=np.int32)
        for wb_file in wb_filenames:
            self._load_file(os.path.join(weights_dir, wb_file))

    def _predict_once(self, inputs, verbose=False):
        '''
        Prediction function for a single window of input from
        one dancer.
        Verbose option is given for analysis and evaluation of model,
        includes stats like time taken for function and output probabilities
        from the model.
        '''
        func_start_time = time.time()

        self.ip_buf = self.xlnk.cma_array(shape=(self.ip_buf_size,),dtype=np.int32) 
        self.op_buf = self.xlnk.cma_array(shape=(self.op_buf_size,), dtype=np.int32)
        
        i = 0
        #print("Weights in stream")
        for element in np.nditer(self.w_arr):
            self.ip_buf[i] = element
            i += 1
            
        assert(i == self.ip_buf_index)
        #print("Inputs in stream")
        for element in np.nditer(inputs):
            self.ip_buf[i] = element
            i += 1
        
        predict_start_time = time.time()

        #print("Sending buffers") 
        self.dma.sendchannel.transfer(self.ip_buf)
        self.dma.recvchannel.transfer(self.op_buf) 
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        #print("Finished waiting")
        
        end_time = time.time()
        #print("Function time: %s s" % round((end_time - func_start_time),3))
        #print("Predict time: %s s" % round((end_time - predict_start_time),3))

        move_index = np.argmax(self.op_buf)
        move_prob = np.amax(self.op_buf)

        if verbose:
            #Get time duration in seconds
            duration = round(end_time - predict_start_time,2)
            #Get output
            output = self.op_buf
            return moves[move_index], output, duration
        
        return moves[move_index], move_prob

    def _predict_one_set(self, input1, input2, input3):
        '''
        Takes 3 inputs from 3 separate dancers, then compares 
        their relative probabilities (in int) and takes the maximum one
        as the move to output, along with its probability
        '''
        move1, move1_p = self._predict_once(input1)
        move2, move2_p = self._predict_once(input2)
        move3, move3_p = self._predict_once(input3)
        
        #Take the dominant move if any two or all three moves match
        if (move1 == move2):
            return move1, max(move1_p, move2_p)
        elif (move2 == move3):
            return move2, max(move2_p, move3_p)
        elif (move1 == move3):
            return move1, max(move1_p, move3_p)
        
        #If all three moves are different, take the one with the highest prob
        p_mv = {move1_p:move1, move2_p:move2, move3_p:move3}
        max_p = max(move1_p, move2_p, move3_p)
        return p_mv[max_p], max_p

    def predict(self, d1_i1, d2_i1, d3_i1, d1_i2, d2_i2, d3_i2, d1_i3, d2_i3, d3_i3):
        '''
        Main prediction function. Takes 3 sets of 3 inputs from
        3 separate dancers for total 9 inputs, then compares their relative
        probabilities (in int) and takes the maximum one
        as the move to output
        '''
        move1, move1_p = self._predict_one_set(d1_i1, d2_i1, d3_i1)
        move2, move2_p = self._predict_one_set(d1_i2, d2_i2, d3_i2)
        move3, move3_p = self._predict_one_set(d1_i3, d2_i3, d3_i3)
        
        #Take the dominant move if any two or all three moves match
        if (move1 == move2):
            return move1
        elif (move2 == move3):
            return move2
        elif (move1 == move3):
            return move1
        
        #If all three moves are different, take the one with the highest prob
        p_mv = {move1_p:move1, move2_p:move2, move3_p:move3}
        return p_mv[max(move1_p, move2_p, move3_p)]

    def dummy_predict(self, inputs):
        '''
        Randomised prediction function
        '''
        self.ip_buf = self.xlnk.cma_array(shape=(self.ip_buf_size,),dtype=np.float32)
        i = 0
        for element in np.nditer(self.w_arr):
            self.ip_buf[i] = element
            i += 1

        assert(i == self.ip_buf_index)
        for element in np.nditer(inputs):
            self.ip_buf[i] = element
            i += 1
        
        classes = ["gun","hair","sidepump"]
        output = randint(0,3)
        return classes[output]

if __name__ == '__main__':
    #test()
    main()
