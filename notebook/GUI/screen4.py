from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import logging

# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.



# Declare both screens
class MenuScreen(Screen):
    pass

class SelectImageScreen(Screen):
    def __init__(self, enterScreen, **kwargs):
        super(SelectImageScreen, self).__init__(**kwargs)
        self.file_chooser.path = "A:\\Testing AI"
        self.enterScreen = enterScreen
        logging.info('in init')
        
    def selected(self,filename):
        try:
            #ids:dictionary of ids
            self.ids.image1.source = filename[0]
            self.enterScreen.ids.image2.source = filename[0]
            logging.info('in try')
        except Exception as e:
            logging.info(e)
            
            
class EnterQuestionScreen(Screen):
    pass
    

class TestApp(App):
    # Create the screen manager
    sm = ScreenManager()
    def build(self):
        Builder.load_file("screen4.kv")
        TestApp.sm.add_widget(MenuScreen(name='menu'))
        enter_screen = EnterQuestionScreen(name = 'enterqns')
        TestApp.sm.add_widget(SelectImageScreen(enter_screen, name='selectimg'))
        TestApp.sm.add_widget(enter_screen)
        return TestApp.sm

if __name__ == '__main__':
    TestApp().run()