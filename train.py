import tensorflow as tf
from keras.applications import VGG16
from keras import layers, models
from tqdm import tqdm

# Prepare the dataset using tf.data.Dataset
def prepare_dataset(data_dir, batch_size):
    print("Preparing the dataset...")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical',
        labels='inferred',
        validation_split=0.2,
        subset='training',
        seed=42
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical',
        labels='inferred',
        validation_split=0.2,
        subset='validation',
        seed=42
    )

    print("Dataset preparation complete.")
    return train_dataset, validation_dataset

# Build the VGG-16 based model
def build_model(num_classes):
    print("Building the model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.7)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    print("Model building complete.")
    return model

# Number of classes (4 rotations: 0, 90, 180, 270)
num_classes = 4

# Build the model
model = build_model(num_classes)

# Compile and train the model
def train_model(model, train_dataset, validation_dataset, num_epochs):
    print("Compiling the model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compilation complete.")

    print("Training the model...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.fit(train_dataset, validation_data=validation_dataset, epochs=1)

    print("Model training complete.")

    # Save the trained model after training is complete
    model.save("Model")
    print("Model saved.")


# Specify the paths
data_dir = "Dataset/Rotated Images"
batch_size = 32

# Prepare the dataset
train_dataset, validation_dataset = prepare_dataset(data_dir, batch_size)

# Training parameters
num_epochs = 10

# Train the model
train_model(model, train_dataset, validation_dataset, num_epochs)