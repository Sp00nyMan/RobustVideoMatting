import os.path

import cv2
import numpy as np

from jnius import autoclass

File = autoclass('java.io.File')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
Tensor = autoclass('org.tensorflow.lite.Tensor')
DataType = autoclass('org.tensorflow.lite.DataType')
TensorBuffer = autoclass(
    'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
ByteBuffer = autoclass('java.nio.ByteBuffer')


class Model:
    MODELS_DIR = 'models/'

    def __init__(self, mode='online', num_threads=None):
        assert mode in ['online', 'local']

        self.model: Interpreter = None
        self.rec = [0.] * 4
        self.downsample_ratio = 1.

        self.num_threads = num_threads
        self.online = mode == 'online'
        if self.online:
            self.__initialize_online()
        else:
            self.__initialize_local()

    def __initialize_local(self):
        model_path = os.path.join(os.getcwd(), self.MODELS_DIR, 'model.tflite')
        model_path = File(model_path)
        options = InterpreterOptions()
        if self.num_threads:
            options.setNumThreads(self.num_threads)
        self.model = Interpreter(model_path, options)
        self._allocate_tensors()

    def _allocate_tensors(self):
        self.model.allocateTensors()
        self.input_shape = self.model.getInputTensor(0).shape()
        self.output_shape = self.model.getOutputTensor(0).shape()
        self.output_dtype = self.model.getOutputTensor(0).dataType()
        print(f'input {self.input_shape}, out_sh {self.output_shape}, out_dt {self.output_dtype}')

    # TODO: Online inference
    def __initialize_online(self):
        pass

    # def _set_input_tensor(self, image):
    #     tensor_index = self.model.get_input_details()[0]['index']
    #     input_tensor = self.model.tensor(tensor_index)[0]
    #
    #     input_tensor[0] = image
    #     input_tensor[1:5] = self.rec
    #     input_tensor[5] = self.downsample_ratio

    def process(self, input_image):
        image: np.ndarray = self._preprocess(input_image)
        input = image.tobytes()
        input = ByteBuffer.wrap(input)
        output = TensorBuffer.createFixedSize(self.output_shape, self.output_dtype)

        print('Model:')
        print(self.model)
        print('IO Tensors:')
        print(input)
        self.model.run(input, output.getBuffer().rewind())
        print(output)

        return self._postprocess(output)

    def _preprocess(self, input_image: np.ndarray):
        if input_image.shape == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        input_image = np.expand_dims(input_image, 0)
        input_image = input_image.astype('float32')
        input_image = np.divide(input_image, 255)
        return input_image

    def _postprocess(self, out):
        fgr, pha, *self.rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']
        result = np.multiply(fgr, pha)
        result = np.squeeze(result, 0)
        result = np.cast(result, 'float32')
        return result
