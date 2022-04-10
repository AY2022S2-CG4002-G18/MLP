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

struct AXIS_wLAST{
	float data;
	bool last;
};

// top level function declaration
void MLP2(hls::stream<AXIS_wLAST>& input_stream, hls::stream<AXIS_wLAST>& output_stream);

int main() {
	std::cout << "start test\n";
	// manages test bench IO with function under test
	AXIS_wLAST write_stream;
	AXIS_wLAST read_stream;
	hls::stream<AXIS_wLAST> input_stream;
	hls::stream<AXIS_wLAST> output_stream;

	// files containing inputs and labels
	std::ifstream input_file;
	input_file.open("test_data.dat");
	std::ifstream label_file;
	label_file.open("test_label.dat");

	// define input and output array
	float input[192];
	float output[4];


	std::string line;
	int test_count = 0;
	int correct = 0;
	int total = 0;
	while(std::getline(input_file, line) && test_count < 10000){
		test_count ++;
		total ++;
		std::stringstream ss(line);
		float val;
		for (int i = 0; i < 192; i ++){
			ss >> val;
			input[i] = val;
		}

		// put data into input, input size = 192
		for (int i = 0; i < 192; i ++){
			write_stream.data = input[i];
			write_stream.last = 0;
			if (i == 192 - 1){
				write_stream.last = 1;
			}
			input_stream.write(write_stream);
		}

		// get output
		MLP2(input_stream, output_stream);

		// output size is 4
		for (int i = 0; i < 4; i ++){
			read_stream = output_stream.read();
			output[i] = read_stream.data;
		}

		// MLE
		int index = 0;
		float max = -INFINITY;
		for (int i = 0; i < 4; i ++){
			if (output[i] > max){
				index = i;
				max = output[i];
			}
		}
		std::cout << "Softmax Array:" << output[0] << "," << output[1] << "," << output[2] << "," <<  output[3] << std::endl;


		// read output and compare
		std::getline(label_file, line);
		int label_value;
		label_value = std::stoi(line);
		std::cout << "Prediction vs Label - " << index << " - " << label_value;
		if (index == label_value){
			std::cout << " [Correct]" << std::endl;
			correct ++;
		} else{
			std::cout << " [incorrect]" << std::endl;
		}
	}

	std::cout << "Correct/Total - " << correct << "/" << total << std::endl;

	return 0;
}
