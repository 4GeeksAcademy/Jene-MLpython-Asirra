# Technical Specifications: Asirra Image Classifier

## 1. Dataset & Preprocessing
* **Scope:** Over 25,000 `.jpg` color images of cats and dogs, with labels parsed directly from filenames.
* **Resizing:** All images are standardized to a fixed spatial dimension of **200x200 pixels**.
* **Memory Management:** 
  * *RAM > 12GB:* Load all images into memory using the Keras image processing API and store as a `(photos, labels)` tuple.
  * *RAM ≤ 12GB:* Stream images progressively using `ImageDataGenerator` and `flow_from_directory()` to accommodate lower-spec hardware.

## 2. Model Architecture
The system utilizes a VGG16-inspired Convolutional Neural Network (CNN) configured for binary classification. The full VGG16 layer setup (including convolutional blocks, max-pooling layers, flattening, and dense layers ending in a 2-unit softmax output) can be found in the referenced model implementation details.

## 3. Optimization, Training & Serialization
* **Callbacks:** Implement `ModelCheckpoint` to save the best performing model weights and `EarlyStopping` to prevent overfitting.
* **Execution:** Train via `.fit_generator()`, evaluate against a test set, and serialize the final model into the target directory.
