from scipy import signal
import tensorflow as tf
import os


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
        dataset_dict[k] = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(data_dict[k], dtype=tf.float32))
        if batch_size is not None:
            dataset_dict[k] = dataset_dict[k].batch(batch_size, drop_remainder=True)

    return dataset_dict


def dict_to_dataset(data_dict, batch_size=None):
    '''
    :param data_dict: Dict with keys state, action, reward, state_new, not_done
    :param batch_size: None, int32 as batch size
    :return: a batched tf.Dataset with the input keys.
    '''
    datasets = [tf.data.Dataset.from_tensor_slices(data_dict[k]) for k in data_dict.keys()]
    ds = tf.data.Dataset.zip(tuple(datasets))

    try:
        ds = ds.map(
            # We want to convert all float64 to float32 to prevent warnings
            lambda state, action, reward, state_new, not_done: (
                tf.cast(state, tf.float32),
                # as we need the action batch to be usable as mask + we need int32s for that.
                tf.expand_dims(tf.cast(action, tf.int32), axis=0),
                tf.cast(reward, tf.float32),
                tf.cast(state_new, tf.float32),
                not_done
            )
        )
    except:
        print("Warning in utils.dict_to_dataset(): Dict keys are not ordered or given as expected.")

    if not batch_size is None:
        ds = ds.batch(batch_size)
    return ds


def all_subdirs_of(b="."):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
