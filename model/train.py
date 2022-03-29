from trainer import Trainer

from dataset import get_data_frame, read_capstone_data

wisdm_df = get_data_frame('wisdm_dataset/WISDM_ar_v1.1_raw.txt')
capstone_df = read_capstone_data('data/data.csv')
capstone_label = read_capstone_data('data/label.csv')

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
trainer.normalize_train()
trainer.initialise_train()
trainer.normalize_test()
trainer.initialise_test()
trainer.create_model()
trainer.train()
# trainer.visualize_training_result()
trainer.visualize_testing_result()
