from trainer import Trainer

from dataset import get_data_frame

df = get_data_frame('wisdm_dataset/WISDM_ar_v1.1_raw.txt')
trainer = Trainer(df)
trainer.train_test_split()
trainer.normalize_train()
trainer.initialise_train()
trainer.normalize_test()
trainer.initialise_test()
trainer.create_model()
trainer.train()
trainer.visualize_training_result()
trainer.visualize_testing_result()
