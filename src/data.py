"""데이터 불러오기와 tf.data 변환. (정제는 cleaning.py, 증강은 augment.py)"""
import numpy as np

from .cleaning import standardize_image
from .config import BATCH_SIZE, DRIVE_PATH, IMG_SIZE, SEED


def load_splits(path=DRIVE_PATH):
    """Drive에 저장해 둔 HF DatasetDict를 읽어 (splits, label_names)로 돌려준다.
    splits는 {"train": [레코드...], ...} 형식 (cleaning.py와 같은 형식)."""
    from datasets import load_from_disk
    ds = load_from_disk(path)
    label_names = ds["train"].features["label"].names
    splits = {name: [{"image": ex["image"], "label": ex["label"]} for ex in part]
              for name, part in ds.items()}
    return splits, label_names


def class_counts(splits, label_names):
    """split별 클래스 개수 표."""
    from collections import Counter
    return {n: {label_names[k]: v for k, v in sorted(Counter(r["label"] for r in recs).items())}
            for n, recs in splits.items()}


def to_tf_dataset(records, training=False, batch_size=BATCH_SIZE, img_size=IMG_SIZE, cache=True):
    """레코드 리스트 -> 배치된 tf.data.Dataset. 학습용만 시드 고정 셔플."""
    import tensorflow as tf

    def gen():
        for r in records:
            img = standardize_image(r["image"], size=img_size)
            yield np.asarray(img, dtype=np.uint8), r["label"]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec(shape=(*img_size, 3), dtype=tf.uint8),
                          tf.TensorSpec(shape=(), dtype=tf.int64)),
    )
    if cache:   # 메모리가 부족하면 False
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(records), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
