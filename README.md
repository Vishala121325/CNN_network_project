# 🍔 Food Image Classification using CNN (Food-101 Dataset)

A deep learning project that builds and trains a **Convolutional Neural Network (CNN)** to classify food images into 101 categories using the **Food-101 dataset**, implemented in **TensorFlow** and developed on **Google Colab**.

---

## 🎯 Objective

To build a CNN model capable of classifying food images into 101 different categories using the Food-101 dataset, with a focus on model performance, augmentation, and visualization.

---

## ✨ Key Features

1. **Uses the Food-101 Dataset**
   - Benchmark dataset with **101,000 images** across 101 food categories.
   - Loaded using `tensorflow_datasets` for streamlined access.

2. **TensorFlow-based CNN Model**
   - Custom **CNN architecture** built from scratch using `tf.keras.Sequential()`.

3. **Data Augmentation**
   - Improves generalization with techniques like **flipping**, **rotating**, and **zooming**.

4. **Training and Validation Visualization**
   - Real-time plotting of training/validation **accuracy** and **loss** using `matplotlib`.

5. **Model Evaluation**
   - Assessed using accuracy/loss curves and validation set performance.
   - Visualizes predictions for sample test images.

---

## 🏷️ Tags

`tensorflow` • `cnn` • `image-classification` • `deep-learning`  
`food101` • `google-colab` • `data-augmentation` • `model-evaluation`

---

## 🔧 Core Functions

| Function                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `tfds.load()`             | Loads Food-101 dataset with training/validation splits                      |
| `map()`                   | Resizes and normalizes images for training                                  |
| `tf.keras.Sequential()`   | Defines the CNN model architecture                                          |
| `model.compile()`         | Configures training (optimizer, loss, metrics)                              |
| `model.fit()`             | Trains the model using the processed dataset                                |
| `model.evaluate()`        | Evaluates model performance on validation set                               |
| `model.predict()`         | Predicts class labels for unseen food images                                |
| `plt.plot()`              | Plots accuracy and loss curves for training/validation                      |

---

## 📊 Outcome of the Project

- ✅ Successfully trained a CNN model that classifies food images into **101 distinct categories**
- 📈 Achieved strong performance through **data augmentation** and **model tuning**
- 🍽️ The trained model can be adapted for:
  - A **food recognition app**
  - An **AI-powered restaurant system** for dish identification

---
