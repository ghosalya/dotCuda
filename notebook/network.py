import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torchvision.models as models

'''
# Image Embed (Resnet152) 
'''  

class ImageEmbedding(nn.Module):
    def __init__(self, image_ftrs, pretrained = True): 
        super(ImageEmbedding, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        num_ftrs = self.resnet.fc.in_features #output 1000 from the resnet to be input to take the fc from the resnet output from doc
        self.resnet.fc = nn.Linear(num_ftrs, hidden_size) #numcls - is our own output classes
    
    def forward(self, image):
        image = self.forwardResnet(image)
        image = F.normalize(image, p=2, dim = 3)
        return image
    
    def forwardResnet(self, image):
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)

        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        image = self.layer4(image)
        

'''
# Question Embed (LSTM) 
'''    
class QnsEmbedding(nn.Module):
    def __init__(self,input_size, question_ftrs, num_layers, batch_first = True):
        super(QnsEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size, question_ftrs, num_layers, batch_first = batch_first)
        
    def forward (self, inputs, cache):
        output, (hn, cn) = self.lstm(inputs, cache)
        return hn, cn
'''
# concat two models - LSTM and RESNET152   
'''    

class ConcatNet(nn.Module):
    def __init__(self, vocab_size, word_emb_size, mode = 'train'):
        self.img_channel = ImageEmbedding(image_ftrs, pretrained = True)
        self.qns_channel = QnsEmbedding(input_size, question_ftrs, num_layers, batch_first = True)
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.all = nn.Sequential(nn.Linear(image_ftrs + question_ftrs, 1024), 
                                nn.ReLU(), 
                                nn.Linear(1024, 3000), 
                                nn.Softmax(dim = 0))
#         self.fc1 = nn.Linear(numcl, 1024) 
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(1024, 3000)
#         self.softmax = nn.Softmax(dim = 0)
        
    def forward(self, image, questions):
        image_embed = self.img_channel(image) #returns tensor
        embeds = self
        questions_embed = self.qns_channel(questions)
        added = torch.cat(image_embed,questions_embed)
        output = self.all(added)
        return output
    
    
