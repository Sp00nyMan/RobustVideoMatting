from camera4kivy import Preview
import cv2
import numpy as np
from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture

class CameraReader(Preview):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = None
        print('CameraReader Initialized')
        # self.model = tf.keras.models.load_model('rvm_resnet50_tf')
        # self.model = tf.function(model)

    #TODO Process the image and output the texture
    #TODO FPS Counter
    def analyze_pixels_callback(self, pixels, image_size, image_pos, image_scale, mirror):
        #rgba = np.fromstring(pixels, 'uint8').reshape(image_size[1], image_size[0], 4)
        #rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        print(f'Analyze_Pixels of size {image_size}')
        texture = np.array([255, 0, 0, 255] * (len(pixels) // 8) + [0, 0, 0, 0] * (len(pixels) // 8), dtype='uint8')
        #texture = np.reshape(texture, rgba.shape).astype('uint8')
        self.make_thread_safe(texture.tostring(), image_size)

    @mainthread
    def make_thread_safe(self, texture, size):
        if not self.mask or self.mask.size[:2] != size:
            self.mask = Texture.create(size=size, colorfmt='rgba')
            #self.mask.flip_vertical()
        self.mask.blit_buffer(texture, colorfmt='rgba')

    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        if self.mask:
            print('Rectangle')
            Color(1, 1, 1, 1)
            Rectangle(texture=self.mask, size=tex_size, pos=tex_pos)