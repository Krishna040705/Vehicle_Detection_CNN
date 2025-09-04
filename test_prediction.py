from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load trained model
model_path = 'vehicle_classifier_model.h5'
model = load_model(model_path)

# Test dataset path
test_path = 'E:/Research/split_dataset/test'

# Image preprocessing
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important to keep order for labels
)

# Predict on test data
predictions = model.predict(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size + 1
)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# True labels
true_labels = test_generator.classes

# Ensure matching lengths
predicted_labels = predicted_labels[:len(true_labels)]

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
labels = ['Non-Vehicle', 'Vehicle']

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Extract TP, TN, FP, FN
tn, fp, fn, tp = cm.ravel()

# Print classification report with % format
report = classification_report(true_labels, predicted_labels, target_names=labels, output_dict=True)

print("\n📋 Classification Report (in %):")
for label in labels:
    precision = report[label]['precision'] * 100
    recall = report[label]['recall'] * 100
    f1 = report[label]['f1-score'] * 100
    support = report[label]['support']
    print(f"{label:13} | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-score: {f1:.2f}% | Support: {support}")

accuracy = report['accuracy'] * 100
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")

# Print confusion matrix values
print("\n📌 Confusion Matrix Values:")
print(f"TP (Vehicle correctly predicted): {tp}")
print(f"TN (Non-Vehicle correctly predicted): {tn}")
print(f"FP (Non-Vehicle predicted as Vehicle): {fp}")
print(f"FN (Vehicle predicted as Non-Vehicle): {fn}")

# ❌ Stop execution after generating report
sys.exit("✅ Test prediction and analysis completed. Execution stopped after classification report.")
