#import tempfile
#import os
#import pickle
import random

import numpy as np


from .base_provider import ImagesDataSet, DataProvider

def augment_image(image, pad):
	return image

'''
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped
'''

def augment_all_images(initial_images, pad):
	return initial_images
'''
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images
'''

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
				by_chanels: substract mean of every chanel and divide each
					chanel data by it's standart deviation
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
	def __init__(self, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
	self.one_hot = one_hot


	size = 30
    self._data_shape = (size, size, 3)
	self._imgdir_path = '/pio/scratch/2/i258312/imdb_crop/01'
	self._mat_path = '/pio/scratch/2/i258312/imdb_crop/imdb.mat'	


	images, labels = self.read_data()
	self.train, self.validation, self.test = self.split_data(images, labels)
    
	@property
    def data_shape(self):
        return self._data_shape

    @property
    def n_classes(self):
        return self._n_classes

	def read_data(self):
		return images, labels

	def split_data(self, images, labels):
		pass
