from tensorflow import keras

from trainer import Trainer
from dataset import get_data_frame
from utils import save_2d_weight_to_txt_file, save_2d_weight_to_dat_file, save_1d_weight_to_txt_file, save_1d_weight_to_dat_file

model = keras.models.load_model('MLP_Model')

df = get_data_frame('wisdm_dataset/WISDM_ar_v1.1_raw.txt')
trainer = Trainer(df)
trainer.train_test_split()
trainer.initialise_train()
trainer.initialise_test()
trainer.normalize_test()
trainer.model = model
trainer.visualize_testing_result()

# print(len(model.trainable_variables))
print(model.summary())

