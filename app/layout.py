from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from camera_reader import CameraReader



class AppLayout(FloatLayout):
    camera_reader = ObjectProperty()

Builder.load_string("""
<AppLayout>:
    camera_reader : self.ids.preview
    CameraReader:
        id:preview
        aspect_ratio: '16:9'
    """)