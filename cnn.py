import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Load images and labels
def load_images_and_labels(root_path):
    images = []
    labels = []
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(folder) # Assuming folder names are the labels
    return images, labels

images, labels = load_images_and_labels(root_path)

# Convert labels to numerical values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def visualize_images(images, labels, label_encoder):
    for label in label_encoder.classes_:
        idx = np.where(labels == label_encoder.transform([label]))[0]
        if len(idx) > 0:
            plt.imshow(images[idx[0]], cmap='gray')
            plt.title(label)
            plt.show()

visualize_images(x_train, y_train, label_encoder)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Reshape images for MLP
x_train_mlp = x_train.reshape(x_train.shape[0], -1)
x_test_mlp = x_test.reshape(x_test.shape[0], -1)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(x_train_mlp, y_train)

# Predict and evaluate
y_pred_mlp = mlp.predict(x_test_mlp)
print("MLP Accuracy:", mlp.score(x_test_mlp, y_test))

# Confusion Matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
ConfusionMatrixDisplay(cm_mlp, display_labels=label_encoder.classes_).plot()
plt.tight_layout()
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Reshape images for CNN
x_train_cnn = np.stack(x_train, axis=0).reshape(x_train.shape[0], 256, 256, 1) # Assuming images are 256x256
x_test_cnn = np.stack(x_test, axis=0).reshape(x_test.shape[0], 256, 256, 1)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile and train CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict and evaluate
y_pred_cnn = model.predict_classes(x_test_cnn)
print("CNN Accuracy:", np.mean(y_pred_cnn == y_test))

# Confusion Matrix
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
ConfusionMatrixDisplay(cm_cnn, display_labels=label_encoder.classes_).plot()
plt.tight_layout()
plt.show()
