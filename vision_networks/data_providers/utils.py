from .cifar import Cifar10DataProvider, Cifar100DataProvider, \
	Cifar10AugmentedDataProvider, Cifar100AugmentedDataProvider
from .svhn import SVHNDataProvider
from .imdbwiki import ImdbWikiDataProvider


def get_data_provider_by_name(name, train_params):
	"""Return required data provider class"""
	if name == 'C10':
		return Cifar10DataProvider(**train_params)
	elif name == 'C10+':
		return Cifar10AugmentedDataProvider(**train_params)
	elif name == 'C100':
		return Cifar100DataProvider(**train_params)
	elif name == 'C100+':
		return Cifar100AugmentedDataProvider(**train_params)
	elif name == 'SVHN':
		return SVHNDataProvider(**train_params)
	elif name in ['IMDBWIKI', 'IMDB', 'IMDB-WIKI', 'IMDB_WIKI']:
		return ImdbWikiDataProvider(**train_params)
	else:
		print("Sorry, data provider for `%s` dataset "
			  "was not implemented yet" % name)
		exit()
