from scipy import signal
import tensorflow as tf

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def dict_to_dataset(data_dict, batch_size=None):

    datasets = []
    for k in dict.keys():
        datasets.append(tf.data.Dataset.from_tensor_slices(data_dict[k]))
    dataset = tf.data.Dataset.zip(tuple(datasets))
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    keys = data_dict.keys()

    return dataset, keys

def dict_to_dict_of_datasets(data_dict, batch_size=None):

    dataset_dict = {}
    for k in data_dict.keys():
        dataset_dict[k] = tf.data.Dataset.from_tensor_slices(data_dict[k])
        if batch_size is not None:
            dataset_dict[k] = dataset_dict[k].batch(batch_size)

    return dataset_dict
