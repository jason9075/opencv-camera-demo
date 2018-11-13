import cv2
import dlib
import multiprocessing

IS_USE_DROP = True  # 閱讀下方is_drop 註解


class DetectionHandler:
    def __init__(self, frame):
        self.process = None
        self.current_frame = frame
        self.detector = dlib.get_frontal_face_detector()

    def start(self, frame_queue, det_queue):
        multiprocessing.Process(target=self.detect_frame, args=(frame_queue, det_queue)).start()

    def detect_frame(self, f_queue, d_queue):
        is_drop = False  # 是否跳過這張(因為外部f_queue 為空時 會丟張進來 但是該張並非是最新的frame)
        while True:
            if f_queue.empty():
                continue

            self.current_frame = f_queue.get()
            if is_drop and IS_USE_DROP:
                is_drop = False
                continue

            if self.current_frame is None:
                break

            rects = self.detector(self.current_frame, 1)
            print("Number of faces detected: {}".format(len(rects)))
            d_queue.put(rects)
            is_drop = True


def main():
    rects = []
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    frame_queue = multiprocessing.Queue()
    detect_queue = multiprocessing.Queue()

    handler = DetectionHandler(frame)
    handler.start(frame_queue, detect_queue)

    while ret:
        ret, frame = cap.read()

        if frame_queue.empty():
            frame_queue.put(frame)
        if not detect_queue.empty():
            rects = detect_queue.get()

        for _, rect in enumerate(rects):
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_queue.put(None)
    cv2.destroyAllWindows()
    cv2.waitKey(4)


if __name__ == '__main__':
    main()
