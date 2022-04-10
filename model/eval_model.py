from tensorflow import keras

from trainer import Trainer
from dataset import get_data_frame, read_capstone_data
from utils import save_2d_weight_to_txt_file, save_2d_weight_to_dat_file, save_1d_weight_to_txt_file, save_1d_weight_to_dat_file

from time import time
model = keras.models.load_model('04101415')

wisdm_df = get_data_frame('wisdm_dataset/WISDM_ar_v1.1_raw.txt')
capstone_df = read_capstone_data('data/data_train.csv')
capstone_label = read_capstone_data('data/label_train.csv')

capstone_df_test = read_capstone_data('data/data_test.csv')
capstone_label_test = read_capstone_data('data/label_test.csv')
# print(capstone_df)
trainer = Trainer(
    wisdm_df=wisdm_df, 
    capstone_df=capstone_df, 
    capstone_label=capstone_label,
    capstone_df_test=capstone_df_test,
    capstone_label_test=capstone_label_test
    )
trainer.train_test_split()
trainer.initialise_train()
trainer.initialise_test()

trainer.normalize_test()
trainer.model = model

trainer.visualize_testing_result()

# print(len(model.trainable_variables))
print(model.summary())

