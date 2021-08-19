import numpy as np
import tensorflow as tf

from typing import Union, Sequence

from PIL import Image

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, tf.Tensor, tf.Variable]


def extract_image_patches(image: TensorLike,
                          patch_dims: TensorLike,
                          stride: TensorLike) -> tf.Tensor:
    """Extracts the image patches of dimension, patch_dims,
   with a stride of integer number.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    patch_dims: A tensor of shape `[H_p, W_p]`, where `H_p` and `W_p` are the
      height and width of the patches.
    stride: An integer number that indicates the rowwise movement between
    successive patches

  Returns:
    A tensor of shape `[N, H_p * W_p]`, where `N` is the total number of patches.

  """

    patches = tf.image.extract_patches(images=image,
                                       sizes=[1, patch_dims[0], patch_dims[1], 1],
                                       strides=[1, stride, stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

    squeezed_patches = tf.squeeze(patches, axis=0)

    reshaped_squeezed_patches = tf.reshape(squeezed_patches,
                                           [squeezed_patches.shape[0] *
                                            squeezed_patches.shape[1],
                                            patch_dims[0] * patch_dims[1]])
    return reshaped_squeezed_patches


def reconstruct_image_from_patches(laplacian_level,
                                   channel,
                                   patches: TensorLike,
                                   img_dims: TensorLike,
                                   stride: TensorLike,
                                   dtype=np.float32) -> tf.Tensor:
    """Combines the patches and saves the reconstructed image.

  Args:
    patches: A tensor of shape `[N, H_p, W_p]`, where `N` is the total number
     of patches, `H_p` and `W_p` are the height and width of the patches.
    img_dims: A tensor of shape `[H, W]`, where `H` and `W` are the
      height and width of the reconstructed image.
    stride: An integer number that indicates the rowwise movement between
    successive patches.

  Returns:
    A tensor of shape `[H, W]`, which is the reconstructed image.
    :param dtype:
    :param stride:
    :param img_dims:
    :param patches:
    :param laplacian_level:
    :param tree_level:

  """
    patches = tf.reshape(patches, [patches.shape[0], 3, 3])
    new_image = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)
    merging_map = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)

    count = 0
    patch_dims = [patches.shape[1], patches.shape[2]]
    for i in range(0, img_dims[0] - patch_dims[0] + 1, stride):
        for j in range(0, img_dims[1] - patch_dims[1] + 1, stride):
            new_image[i:i + patch_dims[0], j:j + patch_dims[1]] = new_image[i:i + patch_dims[0], j:j + patch_dims[1]] + \
                                                                  patches[count][:][:]
            merging_map[i:i + patch_dims[0], j:j + patch_dims[1]] += 1
            count += 1

    image_new = np.divide(new_image, merging_map)

    return image_new
