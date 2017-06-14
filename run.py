from vision_networks.models.dense_net import DenseNet
from vision_networks.data_providers.utils import get_data_provider_by_name

train_params_cifar = {
	'batch_size': 64,
	#'n_epochs': 300,
	'n_epochs': 2,
	'initial_learning_rate': 0.1,
	'reduce_lr_epoch_1': 150,  # epochs * 0.5
	'reduce_lr_epoch_2': 225,  # epochs * 0.75
	'validation_set': True,
	'validation_split': None,  # None or float
	'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
	'normalization': 'by_chanels',	# None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
	'batch_size': 64,
	'n_epochs': 40,
	'initial_learning_rate': 0.1,
	'reduce_lr_epoch_1': 20,
	'reduce_lr_epoch_2': 30,
	'validation_set': True,
	'validation_split': None,  # you may set it 6000 as in the paper
	'shuffle': True,  # shuffle dataset every epoch or not
	'normalization': 'divide_255',
}


def get_train_params_by_name(name):
	if name in ['C10', 'C10+', 'C100', 'C100+']:
		return train_params_cifar
	if name == 'SVHN':
		return train_params_svhn

if __name__ == '__main__':
	model_params = {
			'weight_decay': 0.0001,
			'bc_mode': False,
			'should_save_logs': True,
			'keep_prob': 0.8,
			'reduction': 1.0,
			'dataset': 'C10',
			'model_type': 'DenseNet',
			'depth': 40,
			'train': True,
			'should_save_model': True,
			'test': True,
			'renew_logs': True,
			'total_blocks': 3,
			'nesterov_momentum': 0.9,
			'growth_rate': 12
	}
	
	train_params = {
			'reduce_lr_epoch_1': 150,
			'initial_learning_rate': 0.1,
			'validation_split': None,
			'normalization': 'by_chanels',
			'reduce_lr_epoch_2': 225,
			'shuffle': 'every_epoch',
			'validation_set': True,
			'batch_size': 64,
			'n_epochs': 2,
	}

	print("Params:")
	for k, v in model_params.items():
		print("\t%s: %s" % (k, v))
	print("Train params:")
	for k, v in train_params.items():
		print("\t%s: %s" % (k, v))

	print("Prepare training data...")
	data_provider = get_data_provider_by_name(model_params['dataset'], train_params)
	print("Initialize the model..")
	model = DenseNet(data_provider=data_provider, **model_params)
	if model_params['train']:
		print("Data provider train images: ", data_provider.train.num_examples)
		model.train_all_epochs(train_params)
	if model_params['test']:
		if not model_params['train']:
			model.load_model()
		print("Data provider test images: ", data_provider.test.num_examples)
		print("Testing...")
		loss, accuracy = model.test(data_provider.test, batch_size=200)
		print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
