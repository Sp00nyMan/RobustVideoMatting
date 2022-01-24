import argparse
import time
from threading import Thread, Lock

import cv2
import tensorflow as tf

# --------------- Arguments ---------------

parser = argparse.ArgumentParser(description='Inference from web-cam')
#
# parser.add_argument('--model-type',
#                     type=str,
#                     required=True,
#                     choices=['mattingbase', 'mattingrefine'])
# parser.add_argument('--model-backbone',
#                     type=str,
#                     required=True,
#                     choices=['resnet101', 'resnet50', 'mobilenetv2'])
# parser.add_argument('--model-backbone-scale', type=float, default=0.25)
# parser.add_argument('--model-checkpoint', type=str,
#                     required=True)  # pretrained model
# parser.add_argument('--model-refine-mode',
#                     type=str,
#                     default='sampling',
#                     choices=['full', 'sampling', 'thresholding'])
# parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
# parser.add_argument('--model-refine-threshold', type=float, default=0.7)
#
parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution',
                    type=int,
                    nargs=2,
                    metavar=('width', 'height'),
                    default=(1280, 720))
args = parser.parse_args()

# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.success_reading, self.frame = self.capture.read()

        self.read_lock = Lock()

        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read(cv2.IMREAD_UNCHANGED)
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0))
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


# --------------- Main ---------------

width, height = args.resolution
cam = Camera(width=width, height=height)
dsp = Displayer('Matting',
                cam.width,
                cam.height,
                show_info=(not args.hide_fps))

# Load model

model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
model = tf.function(model)
rec = [ tf.constant(0.) ] * 4
downsample_ratio = tf.constant(0.25)


while True:
    src = cam.read()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    src = tf.expand_dims(src, 0)
    src = tf.cast(src, dtype='float32')
    src = tf.divide(src, 255.)
    out = model([src, *rec, downsample_ratio])

    fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']

    res = tf.multiply(fgr, pha)
    res = tf.squeeze(res, 0)
    res = tf.cast(res, dtype='float32').numpy()

    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    key = dsp.step(res)
    if key == ord('b'):
        break
    elif key == ord('q'):
        exit()
