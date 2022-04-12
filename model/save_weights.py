from tensorflow import keras

from trainer import Trainer
from dataset import get_data_frame
from utils import save_2d_weight_to_txt_file, save_2d_weight_to_dat_file, save_1d_weight_to_txt_file, save_1d_weight_to_dat_file

model = keras.models.load_model('04122200(p)')

# print(len(model.trainable_variables[0]))

# df = get_data_frame('wisdm_dataset/WISDM_ar_v1.1_raw.txt')
# trainer = Trainer(df)
# trainer.train_test_split()
# trainer.initialise_train()
# trainer.initialise_test()
# trainer.normalize_test()
# trainer.model = model
# trainer.visualize_testing_result()

# print(model.trainable_variables[6].numpy().shape)
print(model.summary())

# Save weights into C++ Arrays for Copy Pasting
# Layer 1
layer1_weights_object = model.trainable_variables[0]
layer1_weights_array = layer1_weights_object.numpy()
save_2d_weight_to_txt_file(layer1_weights_array, "./weights_files/layer1_weight.txt")
save_2d_weight_to_dat_file(layer1_weights_array, "./weights_files/layer1_weight.dat")
print("Saved Layer 1 weight")

layer1_bias_object = model.trainable_variables[1]
layer1_bias_array = layer1_bias_object.numpy()
save_1d_weight_to_txt_file(layer1_bias_array, "./weights_files/layer1_bias.txt")
save_1d_weight_to_dat_file(layer1_bias_array, "./weights_files/layer1_bias.dat")
print("Saved Layer 1 bias")

## Layer 2
layer2_weights_object = model.trainable_variables[2]
layer2_weights_array = layer2_weights_object.numpy()
save_2d_weight_to_txt_file(layer2_weights_array, "./weights_files/layer2_weight.txt")
save_2d_weight_to_dat_file(layer2_weights_array, "./weights_files/layer2_weight.dat")
print("Saved Layer 2 weight")

layer2_bias_object = model.trainable_variables[3]
layer2_bias_array = layer2_bias_object.numpy()
save_1d_weight_to_txt_file(layer2_bias_array, "./weights_files/layer2_bias.txt")
save_1d_weight_to_dat_file(layer2_bias_array, "./weights_files/layer2_bias.dat")
print("Saved Layer 2 bias")

## Layer 3
layer3_weights_object = model.trainable_variables[4]
layer3_weights_array = layer3_weights_object.numpy()
save_2d_weight_to_txt_file(layer3_weights_array, "./weights_files/layer3_weight.txt")
save_2d_weight_to_dat_file(layer3_weights_array, "./weights_files/layer3_weight.dat")
print("Saved Layer 3 weight")

layer3_bias_object = model.trainable_variables[5]
layer3_bias_array = layer3_bias_object.numpy()
save_1d_weight_to_txt_file(layer3_bias_array, "./weights_files/layer3_bias.txt")
save_1d_weight_to_dat_file(layer3_bias_array, "./weights_files/layer3_bias.dat")
print("Saved Layer 3 bias")

## Layer 4
layer4_weights_object = model.trainable_variables[6]
layer4_weights_array = layer4_weights_object.numpy()
save_2d_weight_to_txt_file(layer4_weights_array, "./weights_files/layer4_weight.txt")
save_2d_weight_to_dat_file(layer4_weights_array, "./weights_files/layer4_weight.dat")
print("Saved Layer 4 weight")

layer4_bias_object = model.trainable_variables[7]
layer4_bias_array = layer4_bias_object.numpy()
save_1d_weight_to_txt_file(layer4_bias_array, "./weights_files/layer4_bias.txt")
save_1d_weight_to_dat_file(layer4_bias_array, "./weights_files/layer4_bias.dat")
print("Saved Layer 4 bias")

## Layer 5
# layer5_weights_object = model.trainable_variables[8]
# layer5_weights_array = layer5_weights_object.numpy()
# save_2d_weight_to_txt_file(layer5_weights_array, "./weights_files/layer5_weight.txt")
# save_2d_weight_to_dat_file(layer5_weights_array, "./weights_files/layer5_weight.dat")
# print("Saved Layer 5 weight")

# layer5_bias_object = model.trainable_variables[9]
# layer5_bias_array = layer5_bias_object.numpy()
# save_1d_weight_to_txt_file(layer5_bias_array, "./weights_files/layer5_bias.txt")
# save_1d_weight_to_dat_file(layer5_bias_array, "./weights_files/layer5_bias.dat")
# print("Saved Layer 5 bias")