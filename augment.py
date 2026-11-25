"""데이터 증강. 프리셋 이름만 바꿔서 증강 강도를 비교 실험할 수 있다.
증강 레이어는 모델 안에 들어가며 학습할 때만 적용된다.
RandomFlip은 기본에서 제외했다 (뇌 MRI의 좌우 방향 정보를 보존하려는 선택)."""

PRESETS = {
    "none":     {},
    "baseline": {"rotation": 0.05, "zoom": 0.1},                          # 기존 노트북과 동일
    "light":    {"rotation": 0.03, "zoom": 0.05},
    "strong":   {"rotation": 0.10, "zoom": 0.20, "translation": 0.10, "contrast": 0.20},
}


def build_augment(preset="baseline", name="augment"):
    import tensorflow as tf
    from tensorflow.keras import layers

    cfg = PRESETS[preset]
    if not cfg:
        return layers.Identity(name=name)
    stack = []
    if "rotation" in cfg:
        stack.append(layers.RandomRotation(cfg["rotation"]))
    if "zoom" in cfg:
        stack.append(layers.RandomZoom(cfg["zoom"]))
    if "translation" in cfg:
        stack.append(layers.RandomTranslation(cfg["translation"], cfg["translation"]))
    if "contrast" in cfg:
        stack.append(layers.RandomContrast(cfg["contrast"]))
    return tf.keras.Sequential(stack, name=name)


def save_preview(image, augment, path, n=8):
    """한 장의 이미지에 증강을 n번 적용한 결과를 격자로 저장 (README용 그림)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    x = tf.convert_to_tensor(np.asarray(image.convert("RGB")), dtype=tf.float32)[None]
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6))
    for ax in axes.flat:
        ax.imshow(np.clip(augment(x, training=True)[0].numpy(), 0, 255).astype("uint8"))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
