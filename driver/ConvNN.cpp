#include "ap_int.h"
#include "ap_fixed.h"
#include <iostream>
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

int QUANT_FACTOR = 2048;


void conv1d_1(nn_t input[N_TIMESTEPS][N_FEATURES],
	nn_t output[CONV1D1_OUTPUT_SIZE][N_CONV1D_KERNELS],
	nn_t kernel[KERNEL_SIZE][N_FEATURES][N_CONV1D_KERNELS],
	nn_t bias[N_CONV1D_KERNELS],
	bool use_bias=true) {

	for (ind_t filter = 0; filter < N_CONV1D_KERNELS; filter++) {
		for (ind_t timestep = 0; timestep < CONV1D1_OUTPUT_SIZE; timestep += CONV_STRIDE) {
			nn_t sum = 0;
			for (ind_t feature = 0; feature < N_FEATURES; feature++) {
				for (ind_t kernel_cell = 0; kernel_cell < KERNEL_SIZE; kernel_cell++) {
					sum += input[timestep+kernel_cell][feature] * kernel[kernel_cell][feature][filter];
				}
			}

			if (use_bias) {
				sum = sum + bias[filter];
			}

			//ReLU Activation Function
			sum = (sum > (nn_t) 0) ? sum : (nn_t) 0;

			//Scale down (Quantisation)
			output[timestep][filter] = sum/QUANT_FACTOR;
		}
	}
}

void conv1d_2(nn_t input[N_CONV1D1_FEAT][N_CONV1D_KERNELS],
	nn_t output[CONV1D2_OUTPUT_SIZE][N_CONV1D_KERNELS],
	nn_t kernel[KERNEL_SIZE][N_CONV1D_KERNELS][N_CONV1D_KERNELS],
	nn_t bias[N_CONV1D_KERNELS],
	bool use_bias=true) {

	for (ind_t filter = 0; filter < N_CONV1D_KERNELS; filter++) {
		//For every timestep
		for (ind_t conv1d1_feat = 0; conv1d1_feat < CONV1D2_OUTPUT_SIZE; conv1d1_feat += CONV_STRIDE) {
			//For every kernel starting from the timestep
			nn_t sum = 0;

			for (ind_t conv1d1_fil = 0; conv1d1_fil < N_CONV1D_KERNELS; conv1d1_fil++) {
				for (ind_t kernel_cell = 0; kernel_cell < KERNEL_SIZE; kernel_cell++) {
					sum += input[conv1d1_feat+kernel_cell][conv1d1_fil] * kernel[kernel_cell][conv1d1_fil][filter];
				}
			}

			if (use_bias) {
				sum = sum + bias[filter];
			}

			//ReLU Activation Function
			sum = (sum > (nn_t) 0) ? sum : (nn_t) 0;

			//Scale down (Quantisation)
			output[conv1d1_feat][filter] = sum/QUANT_FACTOR;
		}
	}
}

void maxpooling1d(nn_t input[N_CONV1D2_FEAT][N_POOL_SETS],
				nn_t output[MAXPOOL_OUTPUT_SIZE][N_POOL_SETS]) {

	for (ind_t filter = 0; filter < N_POOL_SETS; filter++) {
		ind_t op_ind = 0;
		for (ind_t pool = 0; pool < N_CONV1D2_FEAT; pool += POOL_STRIDE) {
			if (input[pool][filter] >= input[pool+POOL_SIZE-1][filter]) {
				output[op_ind][filter] = input[pool][filter];
			} else {
				output[op_ind][filter] = input[pool+POOL_SIZE-1][filter];
			}
			op_ind++;
		}
	}
}

void flatten(nn_t input[N_MAX_CONV_FEAT][N_FLATTEN_SETS],
		nn_t output[N_FLATTEN_SETS*N_MAX_CONV_FEAT]) {

	ind_t output_index = 0;
	for (ind_t feat = 0; feat < N_MAX_CONV_FEAT; feat++) {
		for (ind_t kernel = 0; kernel < N_FLATTEN_SETS; kernel++) {
			output[output_index] = input[feat][kernel];
			output_index++;
		}
	}
}

void dense_1(nn_t input[N_FLATTEN_FEAT],
	nn_t output[N_HIDDEN_UNITS],
	nn_t weights[N_FLATTEN_FEAT][N_HIDDEN_UNITS],
	nn_t bias[N_HIDDEN_UNITS],
	bool use_bias=true) {

	//Dot product
	for (ind_t i = 0; i < N_HIDDEN_UNITS; i++) {
//#pragma HLS pipeline II=2
		nn_t sum = 0;

		for (ind_t j = 0; j < N_FLATTEN_FEAT; j++) {
			sum += weights[j][i]*input[j];
		}

		if (use_bias) {
			sum = sum + bias[i];
		}
		sum = (sum > (nn_t) 0) ? sum : (nn_t) 0;

		//Scale down (Quantisation)
		output[i] = sum/QUANT_FACTOR;
	}
}

void dense_2(nn_t input[N_DENSE1_FEAT],
	nn_t output[N_DANCE_MOVES],
	nn_t weights[N_DENSE1_FEAT][N_DANCE_MOVES],
	nn_t bias[N_DANCE_MOVES],
	bool use_bias=true) {

	//Dot product
	for (ind_t i = 0; i < N_DANCE_MOVES; i++) {
		nn_t sum = 0;

		for (ind_t j = 0; j < N_DENSE1_FEAT; j++) {
			sum += weights[j][i]*input[j];
		}

		if (use_bias) {
			sum = sum + bias[i];
		}

		sum = (sum > (nn_t) 0) ? sum : (nn_t) 0;

		//Scale down (Quantisation)
		output[i] = sum/QUANT_FACTOR;
	}
}


void cnn(hls::stream<axis_tlast>& ip_stream,
		hls::stream<axis_tlast>& op_stream) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS interface axis port=ip_stream
#pragma HLS interface axis port=op_stream

	axis_tlast read_ip, write_op;
    ind_t i, j, k;

	nn_t conv1_weights[KERNEL_SIZE][N_FEATURES][N_CONV1D_KERNELS];
#pragma HLS array_partition variable=conv1_weights block factor=8 dim=3
	nn_t conv1_bias[N_CONV1D_KERNELS];
	nn_t conv2_weights[KERNEL_SIZE][N_CONV1D_KERNELS][N_CONV1D_KERNELS];
#pragma HLS array_partition variable=conv2_weights block factor=8 dim=3
	nn_t conv2_bias[N_CONV1D_KERNELS];
	nn_t dense1_weights[N_FLATTEN_FEAT][N_HIDDEN_UNITS];
#pragma HLS array_partition variable=dense1_weights block factor=8 dim=1
	nn_t dense1_bias[N_HIDDEN_UNITS];
	nn_t dense2_weights[N_DENSE1_FEAT][N_DANCE_MOVES];
	nn_t dense2_bias[N_DANCE_MOVES];

    nn_t input[N_TIMESTEPS][N_FEATURES];
    nn_t output[N_DANCE_MOVES];


	for (i = 0; i < KERNEL_SIZE; i++) {
		for ( j = 0; j < N_FEATURES; j++) {
			for ( k = 0; k < N_CONV1D_KERNELS; k++) {
				read_ip = ip_stream.read();
				conv1_weights[i][j][k] = read_ip.data;
			}
		}
	}

	for (i = 0; i < N_CONV1D_KERNELS; i++) {
		read_ip = ip_stream.read();
		conv1_bias[i] = read_ip.data;
	}

	for (i = 0; i < KERNEL_SIZE; i++) {
		for (j = 0; j < N_CONV1D_KERNELS; j++) {
			for (k = 0; k < N_CONV1D_KERNELS; k++) {
				read_ip = ip_stream.read();
				conv2_weights[i][j][k] = read_ip.data;
			}
		}
	}

	for (i = 0; i < N_CONV1D_KERNELS; i++) {
		read_ip = ip_stream.read();
		conv2_bias[i] = read_ip.data;
	}

	for (i = 0; i < N_FLATTEN_FEAT; i++) {
		for (j = 0; j < N_HIDDEN_UNITS; j++) {
			read_ip = ip_stream.read();
			dense1_weights[i][j] = read_ip.data;
		}
	}

	for (i = 0; i < N_HIDDEN_UNITS; i++) {
		read_ip = ip_stream.read();
		dense1_bias[i] = read_ip.data;
	}

	for (i = 0; i < N_DENSE1_FEAT; i++) {
		for (j = 0; j < N_DANCE_MOVES; j++) {
			read_ip = ip_stream.read();
			dense2_weights[i][j] = read_ip.data;
		}
	}

	for (i = 0; i < N_DANCE_MOVES; i++) {
		read_ip = ip_stream.read();
		dense2_bias[i] = read_ip.data;
	}

    //Read inputs
    for (i = 0; i < N_TIMESTEPS; i++) {
        for (j = 0; j < N_FEATURES; j++) {
            read_ip = ip_stream.read();
            input[i][j] = read_ip.data;
        }
    }

	nn_t conv1_output[CONV1D1_OUTPUT_SIZE][N_CONV1D_KERNELS];
	nn_t conv2_output[CONV1D2_OUTPUT_SIZE][N_CONV1D_KERNELS];
	nn_t maxpool_output[MAXPOOL_OUTPUT_SIZE][N_POOL_SETS];
	nn_t flatten_output[N_FLATTEN_SETS*N_MAX_CONV_FEAT];
	nn_t dense1_output[N_HIDDEN_UNITS];

	conv1d_1(input, conv1_output, conv1_weights, conv1_bias);
	conv1d_2(conv1_output, conv2_output, conv2_weights, conv2_bias);
	maxpooling1d(conv2_output, maxpool_output);
	flatten(maxpool_output, flatten_output);
	dense_1(flatten_output, dense1_output, dense1_weights, dense1_bias);
	dense_2(dense1_output, output, dense2_weights, dense2_bias);

    for (i = 0; i < N_DANCE_MOVES; i++) {
        write_op.data = output[i];
        write_op.last = 0;
        if (i == N_DANCE_MOVES - 1) {
            write_op.last = 1;
        }
        op_stream.write(write_op);
    }

}
