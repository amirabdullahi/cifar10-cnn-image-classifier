# Simple CIFAR-10 CNN Classifier with Test Image Predictions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ===========================
# 1. LOAD & PREPARE DATA
# ===========================
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# ===========================
# 2. SIMPLE CNN MODEL
# ===========================
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ===========================
# 3. TRAINING
# ===========================
def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )
    return history

# ===========================
# 4. PREDICTION VISUALIZER
# ===========================
def show_predictions(model, x_test, y_test, class_names, num_images=10):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i])
        plt.axis('off')
        color = 'green' if y_pred[i] == y_true[i] else 'red'
        plt.title(f"Pred: {class_names[y_pred[i]]}\nTrue: {class_names[y_true[i]]}", color=color)
    plt.tight_layout()
    plt.show()

# ===========================
# 5. EVALUATION
# ===========================
def evaluate_model(model, x_test, y_test, history):
    # Accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Accuracy & loss curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Predictions
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Show some test images and predictions
    show_predictions(model, x_test, y_test, class_names)

# ===========================
# 6. MAIN
# ===========================
def main():
    print("Simple CNN for CIFAR-10")
    print("=" * 30)

    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train samples: {x_train.shape}, Test samples: {x_test.shape}")

    model = create_model()
    print("\nModel Summary:")
    model.summary()

    history = train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test, history)
    print("Training complete.")

if __name__ == "__main__":
    main()
