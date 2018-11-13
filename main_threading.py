import cv2
import dlib
import threading


class DetectionHandler:
    def __init__(self, frame):
        self.rect = []
        self.current_frame = frame

        self.detector = dlib.get_frontal_face_detector()

    def start(self):
        threading.Thread(target=self.detect_frame, daemon=True, args=()).start()

    def set_frame(self, frame):
        self.current_frame = frame

    def get_dects(self):
        return self.rect

    def detect_frame(self):
        while True:
            self.rect = self.detector(self.current_frame, 1)
            print("Number of faces detected: {}".format(len(self.rect)))


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    handler = DetectionHandler(frame)
    handler.start()

    while ret:
        ret, frame = cap.read()
        handler.set_frame(frame)
        dets = handler.get_dects()

        for _, d in enumerate(dets):
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(4)


if __name__ == '__main__':
    main()
