import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Load dataset
data = []
labels = []

categories = ["with_mask", "without_mask"]

for category in categories:
    path = "dataset/" + category
    label = categories.index(category)

    for img in os.listdir(path)[:1000]:  # Limit to 1000 images per category
        try:
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)

            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(label)
        except:
            pass

# Convert to numpy
data = np.array(data) / 255.0
labels = np.array(labels)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Step 3: Build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Step 4: Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 6: Save model
model.save("models/mask_model.h5")

print("Model trained and saved successfully!")