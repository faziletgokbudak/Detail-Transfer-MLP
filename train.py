import argparse
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

from PIL import Image
from model import MLP
from utils import extract_image_patches, reconstruct_image_from_patches

from typing import Union, Sequence

dtype = np.float32
t_dtype = tf.float32
label_dtype = tf.int32

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, tf.Tensor, tf.Variable]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--test_output_path', type=str)
    parser.add_argument('--lr', type=int, default=1e-2)
    parser.add_argument('--num_channel', type=int, default=3)
    parser.add_argument('--patch_size', type=list, default=[3, 3])
    parser.add_argument('--laplacian_level', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    input_path = args.input_path
    input = Image.open(input_path)
    gray_input = np.asarray(input)  # [:,:,0] #/ 255.0np.asarray(input)

    output_path = args.output_path
    output = Image.open(output_path)
    gray_output = np.asarray(output)  # [:,:,0] #/ 255.0np.asarray(output)#

    test_path = args.test_path
    test = Image.open(test_path)
    gray_test = np.asarray(test)  # [:,:,0] #/ 255.0

    gray_input = tf.dtypes.cast(tf.convert_to_tensor(gray_input), dtype=tf.float32)
    tensor_input = tf.expand_dims(gray_input, axis=0)  # extra dim for batches

    gray_output = tf.dtypes.cast(tf.convert_to_tensor(gray_output), dtype=tf.float32)
    tensor_output = tf.expand_dims(gray_output, axis=0)  # extra dim for batches

    gray_test = tf.dtypes.cast(tf.convert_to_tensor(gray_test), dtype=tf.float32)
    tensor_test = tf.expand_dims(gray_test, axis=0)  # extra dim for batches

    laplacian_input = tfg.image.pyramid.split(tensor_input, args.laplacian_level,
                                              name=None)  # creates Laplacian pyramid
    laplacian_output = tfg.image.pyramid.split(tensor_output, args.laplacian_level, name=None)
    laplacian_test = tfg.image.pyramid.split(tensor_test, args.laplacian_level, name=None)

    mixed_laplacian = []
    for i in range(args.laplacian_level):
        color_channels = []
        for c in range(args.num_channel):
            print(tensor_input.shape)
            print(len(laplacian_input))
            print(laplacian_input[i][:, :, :, c].shape)

            input_c = tf.expand_dims(laplacian_input[i][:, :, :, c], axis=-1)  # extra dim for color channel
            output_c = tf.expand_dims(laplacian_output[i][:, :, :, c], axis=-1)
            test_c = tf.expand_dims(laplacian_test[i][:, :, :, c], axis=-1)

            input_patches = extract_image_patches(input_c, args.patch_size, 1)
            output_patches = extract_image_patches(output_c, args.patch_size, 1)
            test_patches = extract_image_patches(test_c, args.patch_size, 1)

            img_width = laplacian_test[i].shape[1]
            img_height = laplacian_test[i].shape[2]

            model = MLP(args.patch_size[0] * args.patch_size[1])  # patch as a vector
            opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
            model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError())

            history = model.fit(
                input_patches,
                output_patches,
                # batch_size=args.batch_size,
                epochs=args.epoch,
            )
            print(history.history)

            print(test_patches.shape)
            test_predicted = model.predict(
                test_patches,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
            )
            print(test_predicted.shape)

            color_channels.append(reconstruct_image_from_patches(i, c, test_predicted, [img_width, img_height], 1))

        # change this part!!!!
        if len(color_channels) == 1:
            test_output = color_channels[0]
        elif len(color_channels) == 3:
            test_output = np.stack((color_channels[0], color_channels[1], color_channels[2]), axis=2)
        elif len(color_channels) == 4:
            test_output = np.stack((color_channels[0], color_channels[1], color_channels[2], color_channels[3]), axis=2)

        test_output = tf.dtypes.cast(tf.convert_to_tensor(test_output), dtype=tf.float32)
        test_output = tf.expand_dims(test_output, axis=0)  # extra dim for batches

        mixed_laplacian.append(test_output)
        print('mixed lap', len(mixed_laplacian))
    mixed_laplacian.append(laplacian_test[-1])

    img = tfg.image.pyramid.merge(mixed_laplacian, name='None')  # reconstructs image from pyramid
    reconstructed_from_laplacian = tf.squeeze(img, axis=0)
    # reconstructed_from_laplacian = tf.squeeze(reconstructed_from_laplacian, axis=-1)
    img_np = np.array(reconstructed_from_laplacian)  # * 255.
    print(img_np)
    cv2.imwrite(args.test_output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
