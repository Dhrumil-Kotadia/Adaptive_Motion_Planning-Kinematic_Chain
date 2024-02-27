import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load your dataset and split into training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'G:/WPI/Motion Planning/Final Project/CNN/Train_Test set/Train_set',
    target_size=(300, 300),
    batch_size=1,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    classes=['Obstacle_2', 'Obstacle_3', 'Obstacle_4', 'Obstacle_5', 'Obstacle_6']  # Add your class names
)

test_generator = test_datagen.flow_from_directory(
    'G:/WPI/Motion Planning/Final Project/CNN/Train_Test set/Test_set',
    target_size=(300, 300),
    batch_size=1,
    class_mode='categorical',
    classes=['Obstacle_2', 'Obstacle_3', 'Obstacle_4', 'Obstacle_5', 'Obstacle_6']  # Add your class names
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

# Create a simple CNN model for three classes
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multiple classes
              metrics=['accuracy'])

# Train the model

history = model.fit(
    train_generator,
    epochs=1,
    validation_data=test_generator,
    #callbacks=[reduce_lr]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict new images
new_image_path = 'Map17.png'  # Replace with the path to new obstacle map that need to be classified/identified
img = tf.keras.preprocessing.image.load_img(new_image_path, target_size=(300, 300))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_index]
print(f'The image belongs to class {predicted_class_name}')
if predicted_class_name == 'Obstacle_2':
    sampling_method = "Uniform"
    n = 10;
    print ("The sampling method suggested is", sampling_method)
    print ("The number of nodes is", n)
elif predicted_class_name == 'Obstacle_3':
    sampling_method = "Bridge"
    n = 50;
    print ("The sampling method suggested is", sampling_method)
    print ("The number of nodes is", n)
elif predicted_class_name == 'Obstacle_4':
    sampling_method = "Gaussian"
    n = 55;
    print ("The sampling method suggested is", sampling_method)
    print ("The number of nodes is", n)
elif predicted_class_name == 'Obstacle_5':
    sampling_method = "Uniform"
    n = 25;
    print ("The sampling method suggested is", sampling_method)
    print ("The number of nodes is", n)
else:
   print("The Obstacle is unique")
  

# Function to get layer type and color
def get_layer_info(layer):
    if 'Conv2D' in str(layer):
        return 'Conv2D', 'blue'
    elif 'MaxPooling2D' in str(layer):
        return 'MaxPooling2D', 'green'
    else:
        return 'Other', 'red'

# Get the model layers and their types
layers_info = [get_layer_info(layer) for layer in model.layers]

# Plot the model architecture with colored layers
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

# Read the image and display with colored legends
img = plt.imread('model_plot.png')
plt.imshow(img)

# Display colored legends for layer types
for layer_type, color in set(layers_info):
    plt.plot([], [], color=color, label=layer_type)

plt.legend()
plt.show()








