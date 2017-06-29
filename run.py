from vision_networks.models.dense_net import DenseNet
from vision_networks.data_providers.imdbwiki import ImdbWikiAgeDataProvider, ImdbWikiSexDataProvider
#from vision_networks.data_providers.utils import get_data_provider_by_name


if __name__ == '__main__':
	model_params = {
			'weight_decay': 0.0001,
			'bc_mode': False,
			'should_save_logs': True,
			'keep_prob': 0.8,
			'reduction': 1.0,
			'dataset': 'IMDB',
			'model_type': 'DenseNet',
			'depth': 7,
			'train': True,
			'should_save_model': True,
			'test': True,
			'renew_logs': True,
			'total_blocks': 3,
			'nesterov_momentum': 0.9,
			'growth_rate': 12,
			'comment': 'sex-only-48-0.005'
	}
	
	train_params = {
			'reduce_lr_epoch_1': 25,
			'initial_learning_rate': 0.005,
			'validation_split': 0.2,
			'normalization': 'by_channels',
			'reduce_lr_epoch_2': 35,
			'shuffle': None,
			'validation_set': True,
			'batch_size': 64,
			'n_epochs': 50,
			'img_size': '48',
	}

	print("Params:")
	for k, v in model_params.items():
		print("\t%s: %s" % (k, v))
	print("Train params:")
	for k, v in train_params.items():
		print("\t%s: %s" % (k, v))

	print("Prepare training data...")
	data_provider = ImdbWikiSexDataProvider(**train_params)
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
