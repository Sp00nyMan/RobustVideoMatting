import cv2
import numpy as np
from kivy.utils import platform

try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf

  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate

class Model:
    MODELS_DIR = 'models/'
    def __init__(self, mode='online'):
        assert mode in ['online', 'local']

        self.model : Interpreter = None
        self.rec = [0.] * 4
        self.downsample_ratio = 1.

        self.online = mode == 'online'
        if self.online:
            self.__initialize_online()
        else:
            self.__initialize_local()

    def __initialize_local(self):
        model_name = 'model.tflite' if platform == 'android' else 'model_tf'
        self.model = Interpreter(model_path=self.MODELS_DIR + model_name)
        self.model.allocate_tensors()


    def __initialize_online(self):
        pass

    #TODO: Online inference
    def process(self, input_image):
        image = self._preprocess(input_image)
        self._set_input_tensor(image)

        self.model.invoke()

        out = self.model.get_output_tensor()

        return self._postprocess(out)

    def _preprocess(self, input_image : np.ndarray):
        if input_image.shape == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        input_image = np.expand_dims(input_image, 0)
        input_image = np.cast(input_image, 'float32')
        input_image = np.divide(input_image, 255)
        return input_image

    def _set_input_tensor(self, image):
        tensor_index = self.model.get_input_details()[0]['index']
        input_tensor = self.model.tensor(tensor_index)[0]

        input_tensor[0] = image
        input_tensor[1:5] = self.rec
        input_tensor[5] = self.downsample_ratio

    def _postprocess(self, out):
        fgr, pha, *self.rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']
        result = np.multiply(fgr, pha)
        result = np.squeeze(result, 0)
        result = np.cast(result, 'float32')
        return result
