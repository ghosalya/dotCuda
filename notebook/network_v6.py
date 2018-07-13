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
    def __init__(self, mode = 'train', freeze=True, with_attention=True): #1024 or 1000?
        super(ImageEmbedding, self).__init__()
        #get the first 0-7 layers ; delete the avgpool and fc layers
        self.with_attention = with_attention
        resnet = models.resnet152(pretrained=True)
        modules_front = list(resnet.children())[0:8]      
        self.resnet_front = nn.Sequential(*modules_front)
        
        # freezing resnet_front
        if freeze:
            for param in self.resnet_front.parameters():
                param.requires_grad = False
        
        # get the avgpool and fc layers back
        self.resnet_back_pool = nn.AvgPool2d(kernel_size = 10, stride=1, padding=0) #1, 2048, 1, 1
        
    def forward(self, image):
        '''
        Outputs a 10x10x2048
        '''
        image = self.resnet_front(image)
        image = F.normalize(image, p=2, dim=1)
        if not self.with_attention:
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
        inputs_data = self.tanh(inputs.data) 
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
class Attention(nn.Module):
    def __init__ (self, image_ftrs=2048, question_ftrs=1024, k=512, glimpse=2, dropout=True, mode='train'):
        super(Attention, self).__init__()
        self.mode = mode
        self.image_ftrs = image_ftrs
        self.k_size = k
        self.question_ftrs = question_ftrs
        self.glimpse = glimpse
        self.conv_1 = nn.Conv2d(image_ftrs, k, 1) 
        self.question_mid_fc = nn.Linear(question_ftrs, k)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(k, glimpse, 1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward (self, image_embed, questions_embed):
        '''
        image_embed = bx10x10x2048
        questions_embed = bx1x1024
        '''
        b, c, s1, s2 = image_embed.size()
        img_conv = self.conv_1(image_embed)
        
        # tiling
        qns_ft = self.question_mid_fc(questions_embed.view(b, self.question_ftrs))
        tiled_qns_ft = qns_ft.view(b, self.k_size, 1, 1).expand_as(img_conv)
        
        output = self.relu(img_conv + tiled_qns_ft)
        output = self.conv_2(output)
        output = self.softmax(output.view(b, -1)).view(b, self.glimpse, s1, s2)
        return output
    
        
'''
# concat two models - LSTM and RESNET152   
'''    

class ConcatNet(nn.Module):
    def __init__(self, vocab_size, word_emb_size = 300, emb_size = 1024, with_attention=True,
                 glimpse=2, lstm_layers=1, output_size=3000, mode = 'train', freeze_resnet=True):
        super(ConcatNet, self).__init__()
        self.mode = mode
        self.freeze_resnet = freeze_resnet
        self.with_attention = with_attention
        self.glimpse = glimpse

        self.img_channel = ImageEmbedding(mode = mode, freeze=freeze_resnet, with_attention=with_attention)
        self.qns_channel = QnsEmbedding(word_emb_size, question_ftrs=emb_size, num_layers=lstm_layers, batch_first = True)
        if with_attention:
            self.atn_channel = Attention(question_ftrs=emb_size, glimpse=glimpse)
        
        self.word_emb_size = word_emb_size
        #vocab_size: size of dictionary embeddings, word_emb_size: size of each embedding vector
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        if with_attention:
            inner_fc_inputsize = 2048 * self.glimpse + emb_size
        else:
            inner_fc_inputsize = 2048 + emb_size
        self.resolve_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(inner_fc_inputsize, 1024), 
                                nn.ReLU(), 
                                nn.Linear(1024, output_size), 
                                nn.Softmax(dim = 1))
        
    def forward(self, image, questions):
        image_embed = self.img_channel(image) # returns b x 10 x 10 x 2048
        b, c, _, s = image_embed.size()
        emb_qns = self.word_embeddings(questions.data)
        embeds = PackedSequence(emb_qns, questions.batch_sizes)
        
        cache = self.qns_channel.init_cache(batch=questions.batch_sizes[0])
        questions_embed, _ = self.qns_channel(embeds, cache)
        questions_embed = questions_embed[-1]
        
        if self.with_attention:
            img_attn = self.atn_channel(image_embed, questions_embed)
            # combining attention
            image_embed = image_embed.view(b, 1, c, -1).expand(b, self.glimpse, c, 100)
            img_attn = img_attn.view(b, self.glimpse, 1, -1).expand(b, self.glimpse, c, 100)
            image_final = (image_embed * img_attn).sum(dim=3).view(b, -1)
        else:
            image_final = image_embed
        
        added = torch.cat([image_final, questions_embed], dim=1)
        output = self.resolve_fc(added)
        return output
    
    def parameters(self):
        if self.freeze_resnet:
            all_params = [param for param in self.qns_channel.parameters()] \
                         + [param for param in self.resolve_fc.parameters()] \
                         + [param for param in self.word_embeddings.parameters()]
            if self.with_attention:
                all_params += [param for param in self.atn_channel.parameters()]
            return all_params
        else:
            return super(ConcatNet, self).parameters()
    
    
