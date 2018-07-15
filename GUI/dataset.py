import os, copy, random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from build_vocab import *
from build_answers import *
from vqaTools.vqa import VQA
from PIL import Image


class COCODataset(Dataset):
    
    def __init__(self, vocab, answers, rootDir='../../data2', dataSubType='train2014'):
        
        annFile ='{}/v2_mscoco_{}_annotations.json'.format(rootDir, dataSubType)
        quesFile ='{}/v2_OpenEnded_mscoco_{}_questions.json'.format(rootDir, dataSubType)
        self.vqa = VQA(annFile, quesFile)
        self.imgDir = '{}/{}'.format(rootDir, dataSubType)
        self.vocab = vocab
        self.answers = answers
        self.quesIds = self.vqa.getQuesIds()
        self.dataSubType = dataSubType
        self.transform = transforms.Compose([
                         transforms.Resize(299),
                         transforms.CenterCrop(299),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
                         ])
        
    def __len__(self):
        return len(self.quesIds)
        
    def __getitem__(self, index):
        
        """
        returns:
            question: tensor of word-indices
            transformed image: tensor of shape (3, 299, 299)
            answers: tensor of indices mapped to 3000 most frequently occurring answers
            answers not found among 300 most frequently occurring answers are eliminated
        """
        
        quesId = self.quesIds[index]
        
        img_id = self.vqa.qqa[quesId]['image_id'] 
        img_id = str(img_id).zfill(12)
        path = 'COCO_{}_{}.jpg'.format(self.dataSubType, img_id)
#         print(os.path.join(self.imgDir, path))
        image = Image.open(os.path.join(self.imgDir, path)).convert('RGB')

        image = self.transform(image)
            
            
        # Convert question to word ids
        vocab = self.vocab
        question = self.vqa.qqa[quesId]['question']
#         print(question)
        
        tokens = nltk.tokenize.word_tokenize(question.lower())
        question_list = []
        question_list.append(vocab('<start>'))
        question_list.extend([vocab(token) for token in tokens])
        question_list.append(vocab('<end>'))
        question_tensor = torch.Tensor(question_list)
        
        qa = self.vqa.loadQA(quesId)
        
        ans_list = [a['answer'] for a in qa[0]['answers']]
#         print(ans_list)
        
        ans_index_list = [self.answers(ans) for ans in ans_list] #if ans in self.answers.ans2idx.keys()]
        answer_tensor = torch.Tensor(ans_index_list)
        
        return question_tensor, image, answer_tensor     

    def subset(self, fraction=0.5, count=None, shuffle=True):
        '''
        give subset of certain fraction/count
        prioritizes count
        '''
        if not count:
            count = int(len(self.quesIds) * fraction)
        print('Getting subset of length', count, 'out of', len(self))
        subset = copy.deepcopy(self)
        if shuffle: random.shuffle(subset.quesIds)
        subset.quesIds = subset.quesIds[:count]
        return subset
    
#     def __len__(self):
#         return len(self.vqa.dataset['annotations'])
    

def collate(batch):
    """
    args: list of (question, image, answer) tuples
         question: 1D tensor of variable length
         image: tensor of shape (3, 299, 299)
         answer: 1D tensor of variable length
         
    returns:
        question: packed sequence (data: 1D tensor of total questions length, batch_sizes: 1D tensor of max ques length)
        image: tensor of shape (batchsize, 3, 299, 299)
        answer: tuple of 1D tensors of variable length
    """
    # sort batch in descending order by question length
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True) 
    question, image, answer = zip(*sorted_batch)
    
    return torch.nn.utils.rnn.pack_sequence(question), torch.stack(image), torch.stack(answer)