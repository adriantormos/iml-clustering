import numpy as np


def shuffle(dataset):
    np.random.shuffle(dataset)


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def min_max_normalize(array):
   max_value = np.max(array)
   min_value = np.min(array)
   output = np.zeros(array.shape)
   if max_value != min_value:
     output = (array - min_value)/(max_value - min_value)
   return output