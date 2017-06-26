#import tempfile
#import os
#import pickle
import random
import bisect
import numpy as np
import h5py

from .base_provider import ImagesDataSet, DataProvider


class ImdbWikiDataSet(ImagesDataSet):
	def __init__(self, images, labels, n_classes, shuffle, normalization,
				 augmentation):
		"""
		Args:
			images: 4D numpy array
			labels: 2D or 1D numpy array
			n_classes: `int`, number of cifar classes - 10 or 100
			shuffle: `str` or None
				None: no any shuffling
				once_prior_train: shuffle train data only once prior train
				every_epoch: shuffle train data prior every epoch
			normalization: `str` or None
				None: no any normalization
				divide_255: divide all pixels by 255
				divide_256: divide all pixels by 256
				by_channels: substract mean of every channel and divide each
					channel data by it's standart deviation
			augmentation: `bool`
		"""
		if shuffle is None:
			self.shuffle_every_epoch = False
		elif shuffle == 'once_prior_train':
			self.shuffle_every_epoch = False
			images, labels = self.shuffle_images_and_labels(images, labels)
		elif shuffle == 'every_epoch':
			self.shuffle_every_epoch = True
		else:
			raise Exception("Unknown type of shuffling")

		self.images = images
		self.labels = labels
		self.n_classes = n_classes
		self.augmentation = augmentation
		self.normalization = normalization
		self.images = self.normalize_images(images, self.normalization)
		self.start_new_epoch()

	def start_new_epoch(self):
		self._batch_counter = 0
		if self.shuffle_every_epoch:
			images, labels = self.shuffle_images_and_labels(
				self.images, self.labels)
		else:
			images, labels = self.images, self.labels
		if self.augmentation:
			images = augment_all_images(images, pad=4)
		self.epoch_images = images
		self.epoch_labels = labels

	@property
	def num_examples(self):
		return self.labels.shape[0]

	def next_batch(self, batch_size):
		start = self._batch_counter * batch_size
		end = (self._batch_counter + 1) * batch_size
		self._batch_counter += 1
		images_slice = self.epoch_images[start: end]
		labels_slice = self.epoch_labels[start: end]
		if images_slice.shape[0] != batch_size:
			self.start_new_epoch()
			return self.next_batch(batch_size)
		else:
			return images_slice, labels_slice


class ImdbWikiDataProvider(DataProvider):
	def __init__(self, validation_set=False, validation_split=0.2, shuffle=None, normalization=None, **kwargs):
		"""
		Args:
			validation_set: `bool`.
			validation_split: `float` or None
			shuffle: `str` or None
				None: no any shuffling
				once_prior_train: shuffle train data only once prior train
				every_epoch: shuffle train data prior every epoch
			normalization: `str` or None
				None: no any normalization
				divide_255: divide all pixels by 255
				divide_256: divide all pixels by 256
				by_channels: substract mean of every channel and divide each
					channel data by it's standart deviation
		"""
		self.one_hot = True 
		self.normalization = normalization
		self.shuffle = shuffle
		self.data_augmentation = False
		self.validation_split = validation_split
		self.validation_set = validation_set
		self.test_split = 0.2

		self._age_endpoints = 6 * (np.logspace(0, np.log10(120/6+1), 17) - 1)
		self._age_endpoints[-1] = float('inf')
		self._age_classes = zip(self._age_endpoints[:-1], self._age_endpoints[1:])
		self._n_classes = len(self._age_classes) * 2

		self._hdf_path = '/pio/scratch/2/i258312/faces30px.hdf5'
		images, labels = self.read_data(self._hdf_path)
		self._data_shape = images[0].shape
		self.split_data(images, labels)
	
	@property
	def data_shape(self):
		return self._data_shape

	@property
	def n_classes(self):
		return self._n_classes
	
	@property
	def label_length(self):
		return self.n_classes

	def read_data(self, hdf_path):
		f = h5py.File(hdf_path, 'r')
		#img = np.array(f['img'], dtype=np.float32)
		img = f['img']#[:100]
		age = f['age']#[:100]
		sex = f['sex']#[:100]
		
		labels = np.array([ self.encode_age_and_sex(a,s) for (a,s) in zip(age,sex)  ])
		labels = self.labels_to_one_hot(labels)		
	
		return img, np.array(labels, dtype=np.float32)

	def split_data(self, images, labels):
		test_split_idx = int(labels.shape[0] * (1.0 - self.test_split))

		#(images, labels) = self.shuffle_images_and_labels(images, labels)
		if self.validation_set and self.validation_split is not None:
			valid_split_idx = int(test_split_idx * (1.0 - self.validation_split))
			
			self.validation = ImdbWikiDataSet(images=images[valid_split_idx:test_split_idx], labels=labels[valid_split_idx:test_split_idx], shuffle=self.shuffle, n_classes=self.n_classes, normalization=self.normalization, augmentation=self.data_augmentation)
		else:
			valid_split_idx = test_split_idx

		self.train = ImdbWikiDataSet(images=images[:valid_split_idx], labels=labels[:valid_split_idx], shuffle=self.shuffle, n_classes=self.n_classes, normalization=self.normalization, augmentation=self.data_augmentation)

		self.test = ImdbWikiDataSet(images=images[test_split_idx:], labels=labels[test_split_idx:], shuffle=None, n_classes=self.n_classes, normalization=self.normalization, augmentation=False)

	def age2class(self, age):
		return bisect.bisect_right(self._age_endpoints[:-1], age) - 1

	def encode_age_and_sex(self, age, sex):
		return self.age2class(age) * 2 + int(sex)		

	def decode_label(self, code):
		sex = bool(code % 2)
		age_class = int(code // 2)
		return (age_class, sex)

	def shuffle_images_and_labels(self, images, labels):
		rand_indexes = np.random.permutation(images.shape[0])
		shuffled_images = images[rand_indexes]
		shuffled_labels = labels[rand_indexes]
		return shuffled_images, shuffled_labels

