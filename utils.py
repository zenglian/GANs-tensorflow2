import imageio
import tensorflow as tf
import numpy as np
from PIL import Image
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])


def load_mnist_data(batch_size=64, dataset_name="mnist", model_name=None):
    assert dataset_name in ["mnist", "fashion_mnist", "cifar10", "cifar100"], "invalid dataset name: " + dataset_name
    if dataset_name == "mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif "fashion_mnist" == dataset_name:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    # elif datasets=="cifar10":
    #     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # elif datasets=="cifar100":
    #     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
    BUFFER_SIZE = train_images.shape[0]
    if model_name == "WGAN" or model_name == "WGAN_GP":
        train_images = (train_images - 127.5) / 127.5
    else:
        train_images = (train_images) / 255.0
    train_labels = tf.one_hot(train_labels, depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
        batch_size, drop_remainder=True)
    return train_dataset


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    if path.find("WGAN") >= 0 or path.find("WGAN_GP") >= 0:
        image = image * 127.5 + 127.5
    else:
        image = image * 255.0

    return imageio.imwrite(path, tf.cast(image, tf.uint8))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError("in merge(images,size) images parameter ""must have dimensions: HxW or HxWx3 or HxWx4")
