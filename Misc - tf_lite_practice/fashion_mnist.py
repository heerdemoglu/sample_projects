# Tested on Colaboratory with GPU, Tensorflow 2.4.1
# Tested on MacBook Pro M1 with CPU/GPU, Tensorflow 2.4.0

import tensorflow as tf
# work through pip install tensorflow-model-optimization
import tensorflow_model_optimization as tfmot
import numpy as np
import time


def build_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))

    return model


def train(x_train, y_train, x_test, y_test, lr, epochs, batch_size, savedir):
    """
    Train the model given the dataset and the global parameters (LR, EPOCHS and BATCH_SIZE).

    The model is automatically saved after the training.

    """
    model = build_model(x_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )
    print(model.summary())

    start_time = time.time()

    model.fit(
        x=x_train.astype(np.float32),
        y=y_train.astype(np.float32),
        epochs=epochs,
        validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
        batch_size=batch_size,
    )

    end_time = time.time()
    print("Train elapsed time: {} seconds".format(end_time - start_time))

    model.save(savedir, overwrite=True)
    return model


def test(x_test, y_test, loaddir):
    """
    Load any saved model and evaluate it against the test set.
    """
    model = tf.keras.models.load_model(loaddir)
    print(model.summary())

    start_time = time.time()

    scores = model.evaluate(x_test, y_test)

    end_time = time.time()
    print("Test elapsed time: {} seconds".format(end_time - start_time))
    return scores


def pruned_train(x_train, y_train, loaddir, lr, val_split, epochs, batch_size,
                 # prune_summaries,  # uncomment to add prune summary callback directory to save logs.
                 savedir):

    # Load the model:
    model = tf.keras.models.load_model(loaddir)
    num_images = x_train.shape[0] * (1 - val_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning. # ToDo: Further optimization can work good; put these to arguments of method.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.90,
                                                                 begin_step=0,
                                                                 end_step=end_step,
                                                                 frequency=100)
    }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Pruning method requires a recompile.
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )

    # model_for_pruning.summary()   # for debugging, uncomment if you want to inspect.

    # Train for given amount of time.
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # tfmot.sparsity.keras.PruningSummaries(log_dir=prune_summaries),
    ]

    model_for_pruning.fit(x_train, y_train,
                          batch_size=batch_size, epochs=epochs, validation_split=val_split,
                          callbacks=callbacks)

    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    # final_model.summary() # for debugging, uncomment if you want to inspect.

    final_model.save(savedir, overwrite=True)
    return model


def apply_custom_quantization(layer):
    """
    Helper function that quantizes all layers except for batch normalization, and maxpooling
    as these are not supported by TensorFlow 2.4.

    # ToDo: Hacky ways are possible. Different models may work; but more engineering and different network
    #       implementations would be needed.
    # ToDo: eLu quantization is also not accepted by tensorflow; relu is accepted.
    """
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        if not isinstance(layer, tf.keras.layers.MaxPooling2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer


# ToDo: Uses apply_custom_quantization, however this method has bugs listed, commented within.
def quantized_test(x_test, y_test, loaddir):
    """
    Applies quantization to the given model.
    """
    model = tf.keras.models.load_model(loaddir)

    # Proceed with quantization:
    annotated_model = tf.keras.models.clone_model(model, clone_function=apply_custom_quantization)

    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    q_aware_model.summary()

    start_time = time.time()

    q_aware_model.evaluate(x_test, y_test)

    end_time = time.time()
    print("Test elapsed time: {} seconds".format(end_time - start_time))
