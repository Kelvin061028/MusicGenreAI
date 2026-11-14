"""TensorFlow input pipeline for CNN, CRNN, LSTM and ViT music models.

Augmentation is applied on-the-fly to training batches only. The saved .npy
Log-Mel features and validation/test data are never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .config import PROCESSED_DATA_DIR, RANDOM_STATE
except ImportError:
    from config import PROCESSED_DATA_DIR, RANDOM_STATE


def _read_split(split_name: str) -> pd.DataFrame:
    csv_path = PROCESSED_DATA_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path}. Run `python -m src.prepare_dataset` first."
        )
    frame = pd.read_csv(csv_path)
    required = {"feature_path", "label", "genre"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    return frame


def _absolute_feature_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROCESSED_DATA_DIR / path
    return str(path.resolve())


def _load_array(path_tensor: tf.Tensor, label: tf.Tensor):
    def load_numpy(path_bytes):
        path = path_bytes.decode("utf-8")
        array = np.load(path).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D Log-Mel array, got {array.shape} at {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values found in {path}")
        return array

    feature = tf.numpy_function(load_numpy, [path_tensor], Tout=tf.float32)
    feature.set_shape([None, None])
    # Shared image layout: [mel_bins, time_frames, channel].
    feature = feature[..., tf.newaxis]
    return feature, tf.cast(label, tf.int32)


def spec_augment(spectrogram: tf.Tensor,
                 max_frequency_mask: int = 16,
                 max_time_mask: int = 80,
                 num_masks: int = 2) -> tf.Tensor:
    """Apply simple frequency/time masking to one [mel, time, 1] tensor."""
    result = spectrogram
    mel_bins = tf.shape(result)[0]
    time_frames = tf.shape(result)[1]

    for _ in range(num_masks):
        freq_width = tf.random.uniform(
            [], minval=0, maxval=tf.minimum(max_frequency_mask, mel_bins) + 1,
            dtype=tf.int32,
        )
        freq_start = tf.random.uniform(
            [], minval=0, maxval=tf.maximum(mel_bins - freq_width + 1, 1),
            dtype=tf.int32,
        )
        freq_mask = tf.logical_and(
            tf.range(mel_bins) >= freq_start,
            tf.range(mel_bins) < freq_start + freq_width,
        )
        result = tf.where(freq_mask[:, tf.newaxis, tf.newaxis], 0.0, result)

        time_width = tf.random.uniform(
            [], minval=0, maxval=tf.minimum(max_time_mask, time_frames) + 1,
            dtype=tf.int32,
        )
        time_start = tf.random.uniform(
            [], minval=0, maxval=tf.maximum(time_frames - time_width + 1, 1),
            dtype=tf.int32,
        )
        time_mask = tf.logical_and(
            tf.range(time_frames) >= time_start,
            tf.range(time_frames) < time_start + time_width,
        )
        result = tf.where(time_mask[tf.newaxis, :, tf.newaxis], 0.0, result)

    return result


def make_dataset(split_name: str, batch_size: int = 16,
                 model_input: str = "image", augment: bool = False,
                 shuffle: bool = False) -> tf.data.Dataset:
    """Create a dataset.

    model_input='image' -> [batch, mel, time, 1] for CNN/CRNN/ViT.
    model_input='sequence' -> [batch, time, mel] for LSTM.
    Labels are integer class IDs for sparse categorical cross-entropy.
    """
    if model_input not in {"image", "sequence"}:
        raise ValueError("model_input must be 'image' or 'sequence'")
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation or test")

    frame = _read_split(split_name)
    paths = [_absolute_feature_path(p) for p in frame["feature_path"].astype(str)]
    labels = frame["label"].astype("int32").to_numpy()
    if not paths:
        raise ValueError(f"The {split_name} split is empty")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=RANDOM_STATE, reshuffle_each_iteration=True
        )
    dataset = dataset.map(_load_array, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        if split_name != "train":
            raise ValueError("Augmentation is permitted only for the training split")
        dataset = dataset.map(
            lambda x, y: (spec_augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if model_input == "sequence":
        dataset = dataset.map(
            lambda x, y: (tf.transpose(tf.squeeze(x, axis=-1), [1, 0]), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_all_datasets(batch_size: int = 16, model_input: str = "image"):
    """Return train, validation and test datasets with safe augmentation policy."""
    train = make_dataset("train", batch_size, model_input, augment=True, shuffle=True)
    validation = make_dataset("validation", batch_size, model_input, augment=False)
    test = make_dataset("test", batch_size, model_input, augment=False)
    return train, validation, test


if __name__ == "__main__":
    for input_type in ("image", "sequence"):
        train_ds, val_ds, test_ds = load_all_datasets(batch_size=8, model_input=input_type)
        x, y = next(iter(train_ds))
        print(f"\nInput type: {input_type}")
        print("Train batch shape:", x.shape)
        print("Label batch shape:", y.shape)
        print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
        print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
