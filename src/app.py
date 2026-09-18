"""Command-line Asirra cat/dog classifier.

Examples::
	python src/app.py prepare --data data/raw --output data/processed
	python src/app.py train --data data/raw --epochs 20
	python src/app.py predict --model models/asirra.keras --image photo.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from utils import LABELS, ensure_directory_dataset, image_files

IMAGE_SIZE = (200, 200)


def _tensorflow():
	try:
		import tensorflow as tf
	except ImportError as exc:
		raise RuntimeError(
			"TensorFlow is required for train/evaluate/predict. Install requirements.txt."
		) from exc
	return tf


def build_model():
	"""Build the VGG16-inspired binary classifier required by the spec."""
	tf = _tensorflow()
	layers = tf.keras.layers
	model = tf.keras.Sequential([
		layers.Input(shape=(*IMAGE_SIZE, 3)),
		layers.Rescaling(1.0 / 255),
		layers.Conv2D(32, 3, activation="relu", padding="same"),
		layers.Conv2D(32, 3, activation="relu", padding="same"),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, activation="relu", padding="same"),
		layers.Conv2D(64, 3, activation="relu", padding="same"),
		layers.MaxPooling2D(),
		layers.Conv2D(128, 3, activation="relu", padding="same"),
		layers.Conv2D(128, 3, activation="relu", padding="same"),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(256, activation="relu"),
		layers.Dropout(0.5),
		layers.Dense(2, activation="softmax"),
	])
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	return model


def _ram_gb() -> float:
	if platform.system() == "Linux":
		return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
	return 0.0


def prepare_dataset(data: Path, output: Path) -> Path:
	files = image_files(data)
	if not files:
		raise FileNotFoundError(f"No images found under {data}")
	return ensure_directory_dataset(files, output)


def train(args: argparse.Namespace) -> None:
	tf = _tensorflow()
	dataset = prepare_dataset(Path(args.data), Path(args.prepared))
	model_dir = Path(args.model).parent
	model_dir.mkdir(parents=True, exist_ok=True)
	datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		validation_split=args.validation_split,
		rotation_range=15,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
	)
	common = dict(target_size=IMAGE_SIZE, batch_size=args.batch_size, class_mode="categorical", seed=42)
	train_gen = datagen.flow_from_directory(dataset, subset="training", shuffle=True, **common)
	valid_gen = datagen.flow_from_directory(dataset, subset="validation", shuffle=False, **common)
	model = build_model()
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(args.model, monitor="val_accuracy", save_best_only=True, save_weights_only=False),
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
	]
	# fit_generator was removed in recent TensorFlow; use it when available to
	# support the historical API requested by the project specification.
	fit = getattr(model, "fit_generator", model.fit)
	history = fit(train_gen, validation_data=valid_gen, epochs=args.epochs, callbacks=callbacks)
	Path(args.history).parent.mkdir(parents=True, exist_ok=True)
	Path(args.history).write_text(json.dumps({k: [float(v) for v in values] for k, values in history.history.items()}, indent=2))
	print(f"Saved best model to {args.model}; RAM detected: {_ram_gb():.1f} GB")


def evaluate(args: argparse.Namespace) -> None:
	tf = _tensorflow()
	dataset = prepare_dataset(Path(args.data), Path(args.prepared))
	generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
		dataset, target_size=IMAGE_SIZE, batch_size=args.batch_size, class_mode="categorical", shuffle=False
	)
	result = tf.keras.models.load_model(args.model).evaluate(generator, verbose=1, return_dict=True)
	print(json.dumps({key: float(value) for key, value in result.items()}, indent=2))


def predict(args: argparse.Namespace) -> None:
	tf = _tensorflow()
	image = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE)
	values = tf.keras.utils.img_to_array(image)[None, ...] / 255.0
	model = tf.keras.models.load_model(args.model)
	probabilities = model.predict(values, verbose=0)[0]
	index = int(probabilities.argmax())
	print(json.dumps({"label": LABELS[index], "confidence": float(probabilities[index]), "probabilities": dict(zip(LABELS, map(float, probabilities)))}, indent=2))


def main() -> None:
	parser = argparse.ArgumentParser(description="Train and use the Asirra cat/dog classifier")
	sub = parser.add_subparsers(dest="command", required=True)
	def add_common(p):
		p.add_argument("--data", default="data/raw"); p.add_argument("--prepared", default="data/processed/asirra")
		p.add_argument("--batch-size", type=int, default=32)
	p = sub.add_parser("prepare"); p.add_argument("--data", default="data/raw"); p.add_argument("--output", default="data/processed/asirra")
	p.set_defaults(func=lambda a: print(f"Prepared dataset at {prepare_dataset(Path(a.data), Path(a.output))}"))
	p = sub.add_parser("train"); add_common(p); p.add_argument("--model", default="models/asirra.keras"); p.add_argument("--history", default="models/history.json"); p.add_argument("--epochs", type=int, default=20); p.add_argument("--patience", type=int, default=5); p.add_argument("--validation-split", type=float, default=0.2); p.set_defaults(func=train)
	p = sub.add_parser("evaluate"); add_common(p); p.add_argument("--model", default="models/asirra.keras"); p.set_defaults(func=evaluate)
	p = sub.add_parser("predict"); p.add_argument("--model", default="models/asirra.keras"); p.add_argument("--image", required=True); p.set_defaults(func=predict)
	args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
	main()
