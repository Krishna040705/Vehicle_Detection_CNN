import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class VehicleDetectionSystem:
    def __init__(self):
        # Initialize model and parameters
        self.model_path = 'vehicle_classifier_model.h5'
        self.preset = "002"  # Default preset
        self.detection_id = 0  # Will increment with each detection
        self.position = 276  # Example position value
        self.temperature = 1  # Example temperature value
        self.zone = "X08"  # Example zone value

        # Load the model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found! Train the model first.")
        self.model = load_model(self.model_path)

    def process_image(self, img_path):
        """Process an image and return detection result and confidence"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image '{img_path}' not found!")

        # Load and preprocess image
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = self.model.predict(img_array)
        confidence = float(prediction[0][0])

        # Determine result
        is_vehicle = confidence > 0.5
        return is_vehicle, confidence

    def display_detection_info(self, is_vehicle, confidence):
        """Display detection information in the desired format"""
        # Increment ID with each detection
        self.detection_id += 1

        # Format confidence as percentage
        confidence_pct = confidence * 100 if is_vehicle else (1 - confidence) * 100

        # Create the display output
        display_text = [
            f"{40 * ' '}PRESET {self.preset}",
            "",
            f"ID:{self.detection_id:04d}  ",
            f"P:{self.position} T:{self.temperature:03d} Z:{self.zone}",
            "",
            f"{'✅ VEHICLE' if is_vehicle else '❌ NON-VEHICLE'} DETECTED",
            f"Confidence: {confidence_pct:.1f}%"
        ]

        # Print to console (or you could modify this to display in a GUI)
        print("\n".join(display_text))

    def process_and_display(self, img_path):
        """Process an image and display the results"""
        try:
            is_vehicle, confidence = self.process_image(img_path)
            self.display_detection_info(is_vehicle, confidence)
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# Example usage
if __name__ == "__main__":
    detector = VehicleDetectionSystem()

    # You can modify this to process multiple images or use a camera feed
    image_path = "veh000036.png"  # Change to your image path
    detector.process_and_display(image_path)