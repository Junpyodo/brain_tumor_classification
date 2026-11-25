"""평가와 결과 저장. test 평가는 모델/설정을 확정한 뒤 마지막에 한 번만 호출할 것."""
import json
import os

import numpy as np


def evaluate(model, test_ds, label_names, out_dir=None, tag="baseline"):
    """정확도, 클래스별 리포트, 혼동행렬을 계산하고 out_dir에 저장한다."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    loss, acc = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])   # test_ds는 셔플하지 않음
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(cm)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
            json.dump({"test_loss": loss, "test_acc": acc, "report": report,
                       "confusion_matrix": cm.tolist()}, f, indent=2)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(label_names)), label_names, rotation=45, ha="right")
        ax.set_yticks(range(len(label_names)), label_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_confusion_matrix.png"), dpi=120)
        plt.close(fig)
    return {"test_loss": loss, "test_acc": acc, "report": report, "confusion_matrix": cm}


def plot_history(histories, path):
    """여러 단계의 학습 곡선(loss, accuracy)을 이어서 저장."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
    merged = {k: sum((h.history[k] for h in histories), []) for k in keys}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (a, b), title in zip(axes, [("loss", "val_loss"), ("accuracy", "val_accuracy")],
                                 ["loss", "accuracy"]):
        ax.plot(merged[a], label="train")
        ax.plot(merged[b], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)