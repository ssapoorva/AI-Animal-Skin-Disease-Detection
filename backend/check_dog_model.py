import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/dog_model.keras"

TEST_DIR = "../dataset/dog/dog-diseases-datasets/test"

CLASSES = [
    "Dermatitis",
    "Healthy",
    "ringworm"
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\nCLASSIFICATION REPORT")
print("=====================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4
    )
)

print("\nCONFUSION MATRIX")
print("================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\nCLASS ORDER:")
for i, class_name in enumerate(CLASSES):
    print(i, "=", class_name)