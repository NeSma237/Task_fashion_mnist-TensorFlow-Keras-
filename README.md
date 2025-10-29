# 👗 Fashion MNIST Classification using Neural Network

## 🧠 Overview
This project builds and trains a **fully connected neural network** (Multi-Layer Perceptron) using **TensorFlow/Keras** to classify images from the **Fashion MNIST** dataset into 10 different clothing categories such as shirts, trousers, sneakers, and bags.

The goal is to understand how a basic feedforward neural network can learn visual patterns from grayscale images.

---

## 📦 Dataset
The project uses the **Fashion MNIST** dataset, which contains:
- **60,000** training images  
- **10,000** testing images  
- Each image is **28×28 pixels**, grayscale  
- **10 classes** representing clothing types

| Label | Description |
|--------|--------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## ⚙️ Model Architecture
The model is a **Sequential Neural Network** consisting of multiple dense (fully connected) layers with **ReLU** activation and a **Softmax** output layer.

```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(784, activation='relu'))
model.add(Dense(392, activation='relu'))
model.add(Dense(196, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy

---

## 🚀 Training
The model was trained with different configurations to observe how batch size and epochs affect performance:

| Experiment | Epochs | Batch Size |
|-------------|---------|------------|
| Run 1 | 5 | 128 |
| Run 2 | 8 | 64 |
| Run 3 | 20 | 256 |

Training and validation losses were plotted for each experiment to visualize learning behavior and detect potential overfitting.

---

## 📊 Results
- The model successfully learned to classify Fashion MNIST images with **reasonable accuracy**.  
- Training loss decreased steadily across epochs.  
- A small gap between training and validation loss was observed, indicating minor **overfitting**.

Example prediction:

```python
plt.imshow(X_test[0], cmap='gray')
print("Predicted Label:", np.argmax(y_pred[0]))
```

---

## 📈 Visualization
Below is an example of the loss curves for training and validation data:

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

---

## 🔍 Conclusion
This project demonstrates:
1. **Data Loading & Exploration** — Understanding dataset structure and sample visualization.  
2. **Data Preprocessing** — Normalizing images to [0,1] for better convergence.  
3. **Model Building** — Creating a multi-layer perceptron for image classification.  
4. **Training & Evaluation** — Analyzing performance under different hyperparameter settings.  
5. **Prediction & Visualization** — Testing the model and visualizing its predictions.

**Future Improvements:**
- Add **Dropout** layers to reduce overfitting.  
- Experiment with **Convolutional Neural Networks (CNNs)** for higher accuracy.  
- Perform **hyperparameter tuning** for optimal performance.

---

## 🧰 Requirements
Make sure you have the following libraries installed:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

---

## 🪪 Author
👩‍💻 **Nesma Nasser**  
Project: Fashion MNIST Image Classification  
Tools: TensorFlow, Keras, NumPy, Matplotlib  

---

## 📄 License
This project is released under the [MIT License](LICENSE).
