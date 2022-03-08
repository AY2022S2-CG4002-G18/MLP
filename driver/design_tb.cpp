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
void MLP(hls::stream<AXIS_wLAST>& input_stream, hls::stream<AXIS_wLAST>& output_stream);

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

	float input[240] = {
			5.666064739227295, 0.23154591023921967, -4.821603298187256, 1.8387469053268433, -3.296124219894409, 11.604536056518555, -3.1735410690307617, 0.6129156351089478, -7.886181354522705, 3.1463003158569336, -3.023717164993286, -0.08172208815813065, -1.0351464748382568, -2.451662540435791, -1.375655174255371, -2.492523670196533, 1.4573771953582764, -1.4165161848068237, -1.2666922807693481, -3.6774938106536865, 1.8387469053268433, -1.2666922807693481, 2.410801649093628, -0.6129156351089478, 0.04086104407906532, -7.654635429382324, -1.2666922807693481, -5.0122880935668945, 1.9477096796035767, -5.053149223327637, 9.384419441223145, -5.434518814086914, -0.6129156351089478, 4.712640285491943, -1.757024884223938, -0.9125633239746094, 2.6014864444732666, 5.788647651672363, -1.9885708093643188, 1.4165161848068237, 0.4222307801246643, -3.568531036376953, -4.671779155731201, -2.138394594192505, -1.334794044494629, -2.9147543907165527, 3.52767014503479, -4.058863639831543, 7.586533546447754, -5.175732135772705, 1.1849702596664429, -3.7183549404144287, 3.1735410690307617, -1.7297841310501099, 0.8444615602493286, 2.764930486679077, 2.3426997661590576, 4.794362545013428, -1.8387469053268433, 5.366416931152344, 2.1111538410186768, 4.2086873054504395, -5.134871006011963, 3.023717164993286, 2.0702929496765137, -3.1054391860961914, -4.399372100830078, 4.7535014152526855, -0.6946377158164978, -2.8330323696136475, -0.8036004900932312, -3.214401960372925, -2.8738932609558105, -4.4810943603515625, 2.7921712398529053, -4.7535014152526855, -1.5663399696350098, -3.214401960372925, -2.642347574234009, 0.5720546245574951, 3.78645658493042, 0.7627394795417786, 19.572439193725586, -1.116868495941162, 10.079057693481445, 17.07991600036621, 14.015337944030762, 1.116868495941162, 19.572439193725586, 5.243834018707275, -4.331270694732666, 11.917804718017578, 14.818938255310059, 2.5333845615386963, 2.0702929496765137, 19.572439193725586, -5.243834018707275, 9.806650161743164, 14.709975242614746, 3.1735410690307617, 2.764930486679077, 19.31365394592285, -7.654635429382324, 16.3580379486084, 17.502147674560547, 7.818079471588135, 0.7218784093856812, 19.572439193725586, 2.7921712398529053, 1.6480621099472046, 13.443283081054688, 13.211737632751465, 1.8796080350875854, -6.551387310028076, 9.30269718170166, 10.501288414001465, 19.38175392150879, 3.214401960372925, 12.448997497558594, 16.7802677154541, -8.26755142211914, 10.950759887695312, 11.727119445800781, 1.6889231204986572, 2.492523670196533, 19.572439193725586, -3.9499008655548096, 10.038196563720703, 13.334320068359375, 14.178781509399414, 2.1111538410186768, 19.463476181030273, 6.510526180267334, -5.625203609466553, 12.980191230773926, 17.352323532104492, 15.894946098327637, 3.827317714691162, 12.830367088317871, 14.410327911376953, 0.2724069654941559, 11.454712867736816, 9.003049850463867, 7.164302825927734, 3.486809015274048, 4.86246395111084, 19.531578063964844, -3.296124219894409, -7.627394676208496, 6.551387310028076, 10.460427284240723, 3.0645782947540283, 3.5957717895507812, 19.504337310791016, -1.757024884223938, 4.2086873054504395, 3.759216070175171, 2.955615520477295, 19.572439193725586, -1.498238205909729, 0.5311935544013977, 0.7627394795417786, 8.158588409423828, -2.7921712398529053, 13.824652671813965, 1.334794044494629, 5.706925868988037, 2.5606253147125244, 1.9885708093643188, 4.671779155731201, -3.3369851112365723, -7.886181354522705, 4.603677749633789, 3.486809015274048, -0.19068486988544464, 6.469665050506592, -4.372131824493408, 5.747786521911621, 6.210878372192383, 3.78645658493042, -1.757024884223938, 6.319841384887695, -6.129156589508057, 4.944186210632324, 2.5333845615386963, 4.372131824493408, 0.8036004900932312, 5.51624059677124, 2.0702929496765137, 7.627394676208496, 1.0351464748382568, 6.442424297332764, 1.4165161848068237, -6.020193576812744, -6.42880392074585, -0.2724069654941559, 4.44023323059082, 1.1441092491149902, -2.724069595336914, 8.471856117248535, -7.354987621307373, -0.8036004900932312, 0.3813697397708893, 3.52767014503479, -0.34050869941711426, 7.586533546447754, -1.920469045639038, 14.287744522094727, 3.8681788444519043, 5.51624059677124, 2.22011661529541, -0.8036004900932312, 4.2086873054504395, -4.331270694732666, 11.563674926757812, 7.736357688903809, 3.6774938106536865, 0.9942854046821594, -1.5663399696350098, 6.742072105407715, 2.642347574234009, 7.627394676208496, -3.1735410690307617, 0.08172208815813065, 1.757024884223938, -0.04086104407906532, 5.053149223327637, -2.138394594192505, -7.818079471588135, -7.9679036140441895, 0.23154591023921967, 3.255263090133667, 0.108962781727314, 5.979332447052002, 0.19068486988544464, 11.141444206237793, 3.255263090133667, 0.7627394795417786, 8.008764266967773, -0.3813697397708893
	};

	std::string line;
	std::getline(input_file, line);


	std::cout << "Loading Test Data\n";
	// put data into input, input size = 240
	for (int i = 0; i < 240; i ++){
		write_stream.data = input[i];
		write_stream.last = 0;
		if (i == 240 - 1){
			write_stream.last = 1;
		}
		input_stream.write(write_stream);
	}

	MLP(input_stream, output_stream);

	// output size is 6
	for (int i = 0; i < 6; i ++){
		read_stream = output_stream.read();
		std::cout << "Data read:" << read_stream.data << "\n";
	}

	return 0;
}
