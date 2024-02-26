import tensorflow as tf

INPUT_SIZE = 416
CLASSES = 5
BATCH_SIZE = 32

DROPOUT_FACTOR = 0.5


def build_feature_extractor(inputs):

    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(
        INPUT_SIZE, INPUT_SIZE, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_FACTOR)(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    return x


def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x


def build_classifier_head(inputs):
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name='classifier_head')(inputs)


def build_regressor_head(inputs):
    return tf.keras.layers.Dense(units='4', name='regressor_head')(inputs)


def build_model(inputs):

    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs=inputs, outputs=[
                           classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'classifier_head': 'categorical_crossentropy',
                        'regressor_head': 'mse'},
                  metrics={'classifier_head': 'accuracy', 'regressor_head': 'mse'})

    return model


def get_model():
    return build_model(tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1,)))
