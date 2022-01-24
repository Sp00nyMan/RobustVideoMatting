from kivy.config import Config
Config.set('modules', 'monitor', '')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import platform
from kivy.config import Config

Config.set('modules', 'monitor', '')

from layout import AppLayout
if platform == 'android':
    from jnius import autoclass
    from android.permissions import request_permissions, check_permission, Permission
    from android.runnable import run_on_ui_thread
    from android import mActivity

    View = autoclass('android.view.View')

    @run_on_ui_thread
    def hide_landscape_status_bar(instance, width, height):
        # width,height gives false layout events, on pinch/spread
        # so use Window.width and Window.height
        if Window.width > Window.height:
            # Hide status bar
            option = View.SYSTEM_UI_FLAG_FULLSCREEN
        else:
            # Show status bar
            option = View.SYSTEM_UI_FLAG_VISIBLE
        mActivity.getWindow().getDecorView().setSystemUiVisibility(option)
else:
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

class MyApp(App):
    def build(self):
        self.started = False
        if platform == 'android':
            Window.bind(on_resize=hide_landscape_status_bar)
            request_permissions([Permission.CAMERA, Permission.INTERNET])
            print('Permissions requested')
        self.layout = AppLayout()
        print('Build Successful!')
        return self.layout

    def connect_camera(self, permissions = [], grants = []):
        permission = check_permission(Permission.CAMERA) if platform == 'android' \
            else True

        if not permission:
            raise RuntimeError('Camera permission denied!')
        self.layout.camera_reader.connect_camera(enable_analyze_pixels=False, analyze_pixels_resolution=320)

    def on_start(self):
        Clock.schedule_once(self.connect_camera)

    def on_stop(self):
        self.layout.camera_reader.disconnect_camera()

if __name__ == '__main__':

    MyApp().run()