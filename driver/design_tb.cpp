#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <sstream>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"


#include "header/definitions.h"

typedef ap_uint<INDEX_WIDTH> ind_t;
typedef int nn_t;

struct axis_tlast {
	int data;
	ap_uint<1> last;
};

using namespace std;

nn_t nn_t_abs(nn_t number) {
	if (number < 0) {
		return (0-number);
	} else {
		return number;
	}
}

nn_t ERROR_PRECISION = 200;

void load_file(hls::stream<axis_tlast>& ip_stream, axis_tlast& write_strm, ifstream& file, string& line, int& count) {
	  while (getline(file, line)){
	      stringstream ss(line);
	      float val;

	      while (ss >> val) {
	    	  write_strm.data = val;
	    	  cout << "val " << val << ", strm " << write_strm.data << endl;
	    	  //cout << val << endl;
	    	  write_strm.last = 0;
	    	  ip_stream.write(write_strm);
	          if (ss.peek() == ',') ss.ignore();

	          count++;
	      }
	  }

	  cout << "Count " << count << endl;
}


//Test function declaration
void cnn(hls::stream<axis_tlast>& ip_stream,
		hls::stream<axis_tlast>& op_stream);
    
//string WEIGHTS_DIR = "quantized_wb_512\\";
//string OUTPUT_DIR = "layer_outputs\\";
string WEIGHTS_DIR = "full_quantized_wb_2048_2\\";
string OUTPUT_DIR = "full_layer_outputs_4\\";
 
int main()
{
  axis_tlast write_strm;
  axis_tlast read_strm;

  hls::stream<axis_tlast> ip_stream;
  hls::stream<axis_tlast> op_stream;
  hls::stream<axis_tlast> gold_op_stream;
  
  string line;
  
  ifstream c1_weights_file;
  ifstream c2_weights_file;
  ifstream d1_weights_file;
  ifstream d2_weights_file;

  ifstream c1_bias_file;
  ifstream c2_bias_file;
  ifstream d1_bias_file;
  ifstream d2_bias_file;
  
  ifstream input_file;
  ifstream output_file;
  /*
  c1_weights_file.open(WEIGHTS_DIR + "conv1d0_w_q_256.dat");
  c1_bias_file.open(WEIGHTS_DIR + "conv1d0_b_q_256.dat");
  c2_weights_file.open(WEIGHTS_DIR + "conv1d1_w_q_256.dat");
  c2_bias_file.open(WEIGHTS_DIR + "conv1d1_b_q_256.dat");
  d1_weights_file.open(WEIGHTS_DIR + "dense4_w_q_256.dat");
  d1_bias_file.open(WEIGHTS_DIR + "dense4_b_q_256.dat");
  d2_weights_file.open(WEIGHTS_DIR + "dense5_w_q_256.dat");
  d2_bias_file.open(WEIGHTS_DIR + "dense5_b_q_256.dat");
  */

  c1_weights_file.open(WEIGHTS_DIR + "conv1d0_w_q_512.dat");
  c1_bias_file.open(WEIGHTS_DIR + "conv1d0_b_q_512.dat");
  c2_weights_file.open(WEIGHTS_DIR + "conv1d1_w_q_512.dat");
  c2_bias_file.open(WEIGHTS_DIR + "conv1d1_b_q_512.dat");
  d1_weights_file.open(WEIGHTS_DIR + "dense4_w_q_512.dat");
  d1_bias_file.open(WEIGHTS_DIR + "dense4_b_q_512.dat");
  d2_weights_file.open(WEIGHTS_DIR + "dense5_w_q_512.dat");
  d2_bias_file.open(WEIGHTS_DIR + "dense5_b_q_512.dat");
  

  input_file.open("inputs.dat");
  //input_file.open("full_inputs_3.dat");
  output_file.open(OUTPUT_DIR + "dense5_op.dat");
  
  cout << "Loading convolutional weights and biases.." << endl;
  
  int count = 0;

  cout << "\nConv1 weights" << endl;
  load_file(ip_stream, write_strm, c1_weights_file, line, count);
  cout << "\nConv1 bias" << endl;
  load_file(ip_stream, write_strm, c1_bias_file,  line, count);
  cout << "\nConv2 weights" << endl;
  load_file(ip_stream, write_strm, c2_weights_file, line, count);
  cout << "\nConv2 bias" << endl;
  load_file(ip_stream, write_strm, c2_bias_file, line, count);

  cout << "\nDense1 weights" << endl;
  load_file(ip_stream, write_strm, d1_weights_file, line, count);
  cout << "\nDense1 bias" << endl;
  load_file(ip_stream, write_strm, d1_bias_file, line, count);
  cout << "\nDense2 weights" << endl;
  load_file(ip_stream, write_strm, d2_weights_file, line, count);
  cout << "\nDense2 bias" << endl;
  load_file(ip_stream, write_strm, d2_bias_file, line, count);

  cout << "Loading input.." << endl;
  
  //Put data into input  
    
  for (int ip_ind = 0; ip_ind < N_TIMESTEPS; ip_ind++){
      getline(input_file, line);
      stringstream ss(line);
      float val;
       
      for (int feat_ind = 0; feat_ind < N_FEATURES; feat_ind++) {
          
          ss >> val;
          write_strm.data = val;
          //cout << "val " << val << ", strm " << write_strm.data << endl;
          write_strm.last = 0;
          if (feat_ind == N_FEATURES - 1) {
        	  write_strm.last = 1;
          }
          ip_stream.write(write_strm);
          count++;
          if (ss.peek() == ',') ss.ignore();
      }
  }
  cout << "Total count " << count << endl;


  //Call the hardware function
  cout << "Executing function.." << endl;
  cnn(ip_stream, op_stream);


  //Run a software version of the hardware function to validate results
  cout << "Loading golden output.." << endl;
  
  load_file(gold_op_stream, write_strm, output_file, line, count);

  //Compare results

  cout << "Comparing function with golden output.." << endl;
  int mismatch_count = 0;
  nn_t test_val;
  nn_t corr_val;

  for(int comp_1 = 0; comp_1 < N_DANCE_MOVES; comp_1++){

	  read_strm = op_stream.read();
      test_val = read_strm.data;
      if ((comp_1 == N_DANCE_MOVES - 1) && (!read_strm.last)) {
    	  cout << "tlast is not asserted" << endl;
    	  return 1;
      } else {
    	  cout << "TLAST " << read_strm.last << endl;
      }
      read_strm = gold_op_stream.read();
      corr_val = read_strm.data;

      cout << "output at " << comp_1 << ": " << test_val << endl;
      cout << "golden output at " << comp_1 << ": " << corr_val << endl;

      if (nn_t_abs(corr_val - test_val) > ERROR_PRECISION ) {
          //cout << "output at " << comp_1 << ": " << test_val << endl;
          //cout << "golden output at " << comp_1 << ": " << corr_val << endl;

          mismatch_count++;
          //printf("ERROR HW and SW results mismatch\n");
          //return 1;
      }
      if (test_val < 0) {
          cout << "ReLU output negative at " << comp_1 << endl;
          return 1;
      }
  }

  if (mismatch_count) {
	  cout << "Mismatches " << mismatch_count << endl;
	  printf("ERROR HW and SW results mismatch\n");
	  return 1;
  } else {
	  printf("Success HW and SW results match\n");
	  return 0;
  }

}
