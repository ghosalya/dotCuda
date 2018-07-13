import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import PackedSequence

'''
# Image Embed (Resnet152) 
'''  

# COMMENTS

# self.resnet_back is not used

# num_classes = 1000 doesn't exist for our use case:
# we don't have 1000 image classifications, just 1000 arbitrary classes
# this particular fc isn't actually useful because we have 
# another fc layer nn.linear(image_ftrs + question_ftrs, 1024) in ConcatNet

class ImageEmbedding(nn.Module):
    def __init__(self, mode = 'train', freeze=True): #1024 or 1000?
        super(ImageEmbedding, self).__init__()
        #get the first 0-7 layers ; delete the avgpool and fc layers
        resnet = models.resnet152(pretrained=True)
        modules_front = list(resnet.children())[0:8]      
        self.resnet_front = nn.Sequential(*modules_front)
        
        # freezing resnet_front
        if freeze:
            for param in self.resnet_front.parameters():
                param.requires_grad = False
        
        # get the avgpool and fc layers back
        
        modules_avgPool = list(resnet.children())[8:9]
        self.resnet_back_pool = nn.AvgPool2d(kernel_size = 10, stride=1, padding=0) #1, 2048, 1, 1
        
    def forward(self, image):
        image = self.resnet_front(image)
        image = F.normalize(image, p=2, dim = 1)
        image = self.resnet_back_pool(image)
        image = image.view(-1, 2048)
        return image       

'''
# Question Embed (LSTM) 
'''    
class QnsEmbedding(nn.Module):
    def __init__(self, input_size = 300, question_ftrs = 1024, num_layers = 1, batch_first = True): #500 is word embedding size
        super(QnsEmbedding, self).__init__()
        self.tanh = nn.Tanh() #passed into tanh before feed into LSTM
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = question_ftrs, num_layers = num_layers, batch_first = batch_first)
        self.num_layers = num_layers
        self.question_ftrs = question_ftrs
        
    def forward (self, inputs, cache):
        inputs_data = self.tanh(inputs.data) # TODO: incorporate
        inputs = PackedSequence(inputs_data, inputs.batch_sizes)
        output, (hn, cn) = self.lstm(inputs, cache)
        return hn, cn
        
    def init_cache(self, batch=1, use_gpu = True):
        h0 = torch.zeros(self.num_layers, batch, self.question_ftrs)
        c0 = torch.zeros(self.num_layers, batch, self.question_ftrs)
        if use_gpu:
            h0, c0 = h0.cuda(), c0.cuda()
        return (h0, c0)	

'''
# Attention Models 
'''         
class Attention (nn.Module):
    def __init__ (self, output_size = , k = , dropout = True, mode = 'train', freeze_resnet = True):
        super.(Attention, self).__init__()
        self.mode = mode
        self.freeze_resnet = freeze_resnet
        self.img_channel = ImageEmbedding(mode = mode, freeze=freeze_resnet)   
        self.qns_channel = QnsEmbedding(word_emb_size, question_ftrs=emb_size, num_layers=lstm_layers, batch_first = True)
        self.conv_1 = nn.Conv2d(3072, 512, 1) 
        self.relu = nn.ReLu()
        self.conv_2 = nn.Conv2d(512, 2, 1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward (self, output)
        image_embed = self.img_channel(image)
        questions_embed, _ = self.qns_channel(embeds, cache)
        questions_embed = questions_embed[-1]
        concat = torch.cat((image_embed,questions_embed), 1) #concat the img and qns layers
        output = self.conv_1(concat)
        output = self.relu(output)
        output = self.conv_2(output)
        output = self.softmax(output)
        multiply = output * 
        return output
        
'''
# concat two models - LSTM and RESNET152   
'''    

class ConcatNet(nn.Module):
    def __init__(self, vocab_size, word_emb_size = 300, emb_size = 1024, lstm_layers=1, mode = 'train', freeze_resnet=True):
        super(ConcatNet, self).__init__()
        self.mode = mode
        self.freeze_resnet = freeze_resnet
        self.img_channel = ImageEmbedding(mode = mode, freeze=freeze_resnet)
        self.qns_channel = QnsEmbedding(word_emb_size, question_ftrs=emb_size, num_layers=lstm_layers, batch_first = True)
        
        self.word_emb_size = word_emb_size
        #vocab_size: size of dictionary embeddings, word_emb_size: size of each embedding vector
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.resolve_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048 + emb_size, 1024), 
                                nn.ReLU(), 
                                nn.Linear(1024, 3000), 
                                nn.Softmax(dim = 1))
        
    def forward(self, image, questions):
        image_embed = self.img_channel(image) #returns tensor
        emb_qns = self.word_embeddings(questions.data)
        embeds = PackedSequence(emb_qns, questions.batch_sizes)
        cache = self.qns_channel.init_cache(batch=questions.batch_sizes[0])
        questions_embed, _ = self.qns_channel(embeds, cache)
        questions_embed = questions_embed[-1]
        added = torch.cat((image_embed,questions_embed), 1) #concat the img and qns layers
        output = self.resolve_fc(added)
        return output
    
    def parameters(self):
        if self.freeze_resnet:
            return [param for param in self.qns_channel.parameters()] \
                   + [param for param in self.resolve_fc.parameters()] \
                   + [param for param in self.word_embeddings.parameters()]
        else:
            return super(ConcatNet, self).parameters()
    
    
