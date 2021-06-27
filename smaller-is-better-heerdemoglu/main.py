import numpy as np

from fashion_mnist import train, test, pruned_train
import tensorflow as tf

# Global Variables:
LR = 1E-3
EPOCHS = 10
BATCH_SIZE = 64


def main():

    print(tf.config.list_physical_devices())

    # Load the dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Add a trailing unitary dimension to make a 3D multidimensional array (tensor):
    # N x 28 x 28 --> N x 28 x 28 x 1
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert the labels from integers to one-hot encoding:
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Comment/uncomment the following two lines as needed:
    _ = train(x_train, y_train, x_test, y_test, LR, EPOCHS, BATCH_SIZE, 'fashion_mnist_model')

    # Check and list scores of the regular model given with this code segment.
    scores = test(x_test, y_test, './fashion_mnist_model')
    print('Model Validation Loss:', scores[0])
    print('Model Validation Accuracy:', scores[1])

    # Model pruning using TF's optimization libraries. PyTorch port is also available.
    # The support of this API is limited; but allows writing custom pruning and quantization functions.
    pruned_model = pruned_train(x_train, y_train, './fashion_mnist_model', LR, 0.1, 2, BATCH_SIZE,
                                './temp', './pruned_fashion_mnist_model')

    # Observe and evaluate results.
    scores = pruned_model.evaluate(x_test, y_test, verbose=0)

    print('Pruned Validation Loss:', scores[0])
    print('Pruned Validation Accuracy:', scores[1])

    # ToDo: Clustering Step:

    # ToDo: Quantization Step:
    # quantized_test(x_test, y_test, './fashion_mnist_model')

    # ToDo: Encoding Step:

    # ToDo: Migrate to TFLite:


if __name__ == "__main__":
    main()

# ToDo - Pain Point Resolutions:
#        Quantize the network further to reduce model file size.
#        Enhancements: ---------------------------------------------------------------------------------------
#        Use of EfficientNet (or other architectures) which are deeper, but has higher accuracy.
#        Utilize knowledge distillation to prune the network.

# ToDo - Further Applications:
#        Check CoreML implementations, develop PyTorch port for this application (if needed).
#        Check compatibility to mobile platforms.

# Notes:
# Made the methods such a way that it can prune any given h5 or tf model file.
# Further (deeper) methods can be trained on top of this network given with the code segment. Deeper models can
# achieve higher capacities and can be pruned even further. Clustering and quantization can compress the network
# size even more; then this can be converted to tflite model for mobile deployment.

# ToDo - Documentation: (Other file)
#        What was done, what are the results? (Compare deep and pruned models)
#        Explain in the report the network compression process; how libraries are used, what do they do?

# ToDo - Submission:
#        Create a Google Colaboratory notebook submission for ease of access. (In progress)
