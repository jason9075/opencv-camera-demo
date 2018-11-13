import cv2
import dlib


def main():
    cap = cv2.VideoCapture(0)
    cap.read()

    detector = dlib.get_frontal_face_detector()
    while True:
        ret, frame = cap.read()

        dets = detector(frame, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for _, d in enumerate(dets):
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(4)


if __name__ == '__main__':
    main()
