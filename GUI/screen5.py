from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import logging
import nltk
import torch
import pickle
import build_vocab
import build_answers
from network_v6 import ConcatNet
import torchvision.transforms as transforms
from PIL import Image
from dataset import collate
import os


# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.



# Declare both screens
class MenuScreen(Screen):
    pass

class SelectImageScreen(Screen):
    def __init__(self, enterScreen, **kwargs):
        super(SelectImageScreen, self).__init__(**kwargs)
        self.file_chooser.path = os.path.expanduser("~/Documents")
        
        self.enterScreen = enterScreen
        logging.info('in init')
        
    def selected(self,filename):
        try:
            #ids:dictionary of ids
            self.ids.image1.source = filename[0]
            self.enterScreen.ids.image2.source = filename[0]
            logging.info(filename)
            logging.info('in try')
        except Exception as e:
            logging.info(e)
            
            
class EnterQuestionScreen(Screen):
    def __init__(self, **kwargs):
        super(EnterQuestionScreen, self).__init__(**kwargs)
        vocab_path = 'vocab.pkl'
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            self.vocab = vocab
        answers_path = 'answers.pkl'
        with open(answers_path, 'rb') as f:
            answers = pickle.load(f)
            self.answers = answers
        self.transform = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        device = torch.device('cuda')
        vocab_size = len(vocab)
        self.model = ConcatNet(vocab_size, with_attention=True).to(device)

        state_dict = torch.load('default_model.pt')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        
    def changeImg2_Tensor(self):
    
        image = Image.open(self.ids.image2.source).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            logging.info(image)
        return image
            
            
    def changeQns2_Tensor(self, qns):
        self.ids.input.source = qns
        tokens = nltk.tokenize.word_tokenize(qns.lower())
        question_list = []
        question_list.append(self.vocab('<start>'))
        question_list.extend([self.vocab(token) for token in tokens])
        question_list.append(self.vocab('<end>'))
        question_tensor = torch.Tensor(question_list)
        
        logging.info('in changeqns func')
        logging.info(qns)
        logging.info(question_tensor)
        
        return question_tensor
        
    def modelpass(self):

        image = self.changeImg2_Tensor()
        qns = self.changeQns2_Tensor(self.ids.input.text)
        logging.info(image)
        logging.info(qns)
        qns, img, _ = collate([(qns, image, torch.Tensor([0]))])
        output = self.model(img.cuda(), qns.long().cuda())
        confidence, pred = output.topk(10)
        ans = [self.answers.idx2ans[j.item()] for j in pred[0]]
        logging.info(confidence)
        if '<unk>' in ans:
            ans.remove('<unk>')
        logging.info(ans)
        firstAns = 'I think it is ' + ans[0] + ' with ' + str(round(confidence[0][0].item(), 2)) + ' confidence'
        secondPart = '\nBut these are my other predictions: ' + ' or '.join(ans [1:4])
        self.ids.ans.text = firstAns + secondPart
        
class TestApp(App):
    # Create the screen manager
    sm = ScreenManager()
    def build(self):
        Builder.load_file("screen5.kv")
        TestApp.sm.add_widget(MenuScreen(name='menu'))
        enter_screen = EnterQuestionScreen(name = 'enterqns')
        TestApp.sm.add_widget(SelectImageScreen(enter_screen, name='selectimg'))
        TestApp.sm.add_widget(enter_screen)
        return TestApp.sm

if __name__ == '__main__':
    TestApp().run()