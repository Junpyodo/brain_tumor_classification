"""프로젝트 전체에서 공유하는 설정. 값을 바꿀 때는 여기만 수정."""
"""시드, 이미지 크기, 라벨 등의 공통설정"""

import random

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_ID = "PranomVignesh/MRI-Images-of-Brain-Tumor"
DRIVE_PATH = "/content/drive/MyDrive/brain_tumor"   # Colab에서 save_to_disk 한 위치

LABEL_NAMES = ["glioma", "meningioma", "no-tumor", "pituitary"]

# 지저분한 라벨 문자열을 표준 이름으로 매핑 (키는 normalize_name 적용 후의 형태)
LABEL_ALIASES = {
    "notumor": "no-tumor",
    "no-tumour": "no-tumor",
    "normal": "no-tumor",
    "healthy": "no-tumor",
    "glioma-tumor": "glioma",
    "meningioma-tumor": "meningioma",
    "pituitary-tumor": "pituitary",
}


def normalize_name(s):
    """다 소문자. 앞뒤 공백, '_'와 공백을 '-'로 통일."""
    return str(s).strip().lower().replace("_", "-").replace(" ", "-")


def set_seed(seed=SEED):
    """파이썬, numpy, (설치돼 있으면) tensorflow 시드를 한 번에 고정."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
