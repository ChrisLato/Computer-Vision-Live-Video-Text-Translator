import cv2  # OpenCV for video capture
import easyocr  # EasyOCR for Optical Character Recognition (OCR)
from googletrans import Translator  # Google Translate API

"""
Code developed by Chris Latosinsky

python version 3.11.7
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

OpenCV:
pip install opencv-python
pip uninstall opencv-python-headless
pip install opencv-contrib-python

EasyOCR:
pip install easyocr

Google Translate:
pip install googletrans==4.0.0-rc1
"""


# Variables to store bounding boxes, detected text, and translations
bounding_boxes = []
translations = []

# Toggle for language detection ('ch_sim' for Simplified Chinese, 'ru' for Russian, 'zh_tr' for Traditional Chinese)
target_language = 'ch_sim'  


def process_frame(frame, reader, translator):
    global bounding_boxes, translations  

    # Convert frame to RGB for EasyOCR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # EasyOCR to detect text
    results = reader.readtext(rgb_frame)

    # Update bounding boxes and translations
    bounding_boxes = []
    translations = []
    for (bbox, text, prob) in results:
        # Store bounding box coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        bounding_boxes.append((top_left, bottom_right, text))

        # Translate detected text
        try:
            translation = translator.translate(text.strip(), dest="en").text
        except Exception as e:
            translation = "Translation Error"
        translations.append(translation)

    # Print detected text and translations
    for i in range(len(bounding_boxes)):
        print(f"Detected Text: {bounding_boxes[i][2]}, Translation: {translations[i]}")


def draw_bounding_boxes(frame):
    global bounding_boxes, translations 

    # Redraw all saved bounding boxes, detected text, and translations on the current frame
    for i, (top_left, bottom_right, text) in enumerate(bounding_boxes):
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Draw translated text above the bounding box
        translation = translations[i] if i < len(translations) else "Translation Error"
        cv2.putText(frame, translation, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main():
    global bounding_boxes, translations, target_language  

    # Initialize EasyOCR
    reader = easyocr.Reader([target_language, 'en'], gpu=True)

    # Initialize Google Translate
    translator = Translator()

    # Capture live video using webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_count = 0  # Counter to process every Nth frame
    process_every = 30  # Process every Nth frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from the camera.")
            break

        # Only process every Nth frame for OCR
        if frame_count % process_every == 0:
            process_frame(frame, reader, translator)

        # Draw bounding boxes and text on every frame
        draw_bounding_boxes(frame)

        # Display live feed with bounding boxes and translations
        cv2.imshow("Live Feed", frame)

        frame_count += 1

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
