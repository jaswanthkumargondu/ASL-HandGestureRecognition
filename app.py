import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('final_model.h5')

final_ans = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del',
    27: 'nothing', 28: 'space'
}

cap = cv2.VideoCapture(0)

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    roi = frame[50:250, 80:280]  # region of interest
    processed_img = preprocess(roi)

    cv2.rectangle(frame, (80, 50), (280, 250), (0, 255, 0), 2)

    try:
        pred = model.predict(processed_img)
        result = np.argmax(pred, axis=1)[0]
        cv2.putText(frame, final_ans[result], (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    except Exception as e:
        print("An error occurred:", e)

    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
