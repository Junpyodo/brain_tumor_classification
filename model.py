"""모델 구성과 2단계 학습 (1단계: 헤드만, 2단계: 상위 층 파인튜닝)."""
import tensorflow as tf
from tensorflow.keras import layers

from .config import IMG_SIZE


def build_efficientnet(num_classes, augment, img_size=IMG_SIZE, dropout=0.3):
    """EfficientNetB0는 0~255 입력을 받으므로 Rescaling이 필요 없다."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def train_two_stage(model, base, train_ds, val_ds, epochs1=10, epochs2=10,
                    lr1=1e-3, lr2=1e-5, unfreeze_last=20, patience=3):
    """학습된 두 단계의 history를 (h1, h2)로 돌려준다."""
    def stopper():
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)

    _compile(model, lr1)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs1, callbacks=[stopper()])

    base.trainable = True
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False   # BN은 고정
    _compile(model, lr2)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs2, callbacks=[stopper()])
    return h1, h2
