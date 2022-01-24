import os.path

import cv2
import numpy as np
import tensorflow as tf

Interpreter = tf.lite.Interpreter


class Model:
    MODELS_DIR = 'models'

    def __init__(self, mode='online', num_threads=None):
        assert mode in ['online', 'local']

        self.model: Interpreter = None
        self.rec = np.array([0.] * 4).astype('float32')
        self.downsample_ratio = np.float32(1.)

        self.num_threads = num_threads
        self.online = mode == 'online'
        if self.online:
            self.__initialize_online()
        else:
            self.__initialize_local()

    def __initialize_local(self):
        self.model = Interpreter(model_path=os.path.join(os.getcwd(), self.MODELS_DIR, 'model.tflite'),
                                 num_threads=self.num_threads)
        self.model.resize_tensor_input(2, (1, 720, 1280, 3))
        self._allocate_tensors()

    def _allocate_tensors(self):
        self.model.allocate_tensors()

    def __initialize_online(self):
        pass
    # TODO: Online inference

    def _set_input_tensor(self, image):
        [print(x.shape) for x in self.rec]
        print('-----')
        input_details = self.model.get_input_details()

        # need_reallocate = False
        # for i, r in enumerate([3, 0, 4, 1]):
        #     if self.model.tensor(r)().shape != self.rec[i].shape:
        #         self.model.resize_tensor_input(r, self.rec[i].shape)
        #         need_reallocate = True
        # if need_reallocate:
        #     self._allocate_tensors()

        for i, r in enumerate([3, 0, 4, 1]):
            self.model.set_tensor(r, self.rec[i])

        self.model.set_tensor(2, image)
        self.model.set_tensor(5, self.downsample_ratio)


    def _preprocess(self, input_image: np.ndarray):
        if input_image.shape[-1] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        input_image = np.expand_dims(input_image, 0)
        input_image = input_image.astype('float32')
        input_image = np.divide(input_image, 255)
        return input_image

    def process(self, input_image):
        image = self._preprocess(input_image)
        self._set_input_tensor(image)

        self.model.invoke()
        fgr, pha = self._get_output()
        return self._postprocess(fgr, pha)

    def _get_output(self):
        output = []
        output_details = self.model.get_output_details()
        for t in output_details:
            output.append(self.model.get_tensor(t['index']))
        fgr, pha = output[:2]
        self.rec = output[2:]
        return fgr, pha

    def _postprocess(self, fgr, pha):
        result = np.multiply(fgr, pha)
        result = np.squeeze(result, 0)
        result = np.multiply(result, 255)
        return result
