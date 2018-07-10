'''
Trainer class for VQA Data
that can train and test an accepted model
'''
import time
import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class VQATrainer:
    def __init__(self, model, device, accuracy_fn='approve3'):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        accuracy_fn_list = self.get_accuracy_fns()
        self.accuracy_fn = accuracy_fn_list[accuracy_fn]
        self.statistics = {}

    def train(self, train_dataset, val_dataset, epoch=5, learnrate=1e-2):
        '''
        Train over many epoch, outputing test result
        in between
        '''
        sgd_optimizer = torch.optim.SGD(self.model.parameters(), learnrate)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)

        self.statistics = {'train-losses':[],
                           'val-losses': [],
                           'train-accuracy': [],
                           'val_accuracy': []}

        for e in range(epoch):
            # training phase
            print('Epoch {} of {}'.format(e, epoch))
            print('  Training...')
            self.model, train_loss, train_acc = self.train_epoch(train_loader, optimizer=sgd_optimizer)
            self.statistics['train-losses'].append(train_loss)
            self.statistics['train-accuracy'].append(train_acc)
            # validation phase
            print('  Validating...')
            self.model, val_loss, val_acc = self.test_epoch(val_loader)
            self.statistics['val-losses'].append(val_loss)
            self.statistics['val-accuracy'].append(val_acc)

        return self.model, self.statistics

    def train_epoch(self, dataloader, optimizer=None, mode='train', print_every=1):
        '''
        Train function fro one epoch, returning the model, loss & accuracy
        '''
        if mode == 'train' and optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)

        self.model.train(mode == 'train')
        self.model.zero_grad()

        running_loss = 0
        running_correct = 0

        epoch_start = time.clock()
        iterr = 0
        for data in dataloader:
            iterr += 1

            # wrapping data
            image = Variable(data['image'].to(self.device))
            question = Variable(data['question'].to(self.device))
            labels = data['label'].to(self.device)

            # forward pass
            self.model.zero_grad()
            outputs = self.model(image, question)

            # handling loss
            total_loss = None
            for i, label in enumerate(labels): # loop by batch
                one_data_loss = None
                for j in range(label.size().item()):
                    if total_loss is None:
                        one_data_loss = self.criterion(outputs, label[j])
                    else:
                        one_data_loss += self.criterion(outputs, label[j])
                one_data_loss = one_data_loss / label.size()
                # assigning to total loss
                if total_loss is None:
                    total_loss = one_data_loss
                else:
                    total_loss += one_data_loss

            running_loss += total_loss.item()

            # accuracy prediction
            running_correct += self.accuracy_fn(outputs, label)

            # backpropagation
            if mode == 'train':
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            self.print_every(iterr, len(dataloader), print_every)
        
        epoch_end = time.clock()
        accuracy = running_correct / len(dataloader)

        print('   >> Epoch finished with loss {:.5f} and accuracy {:.3f} in {:.4f}s'\
              .format(running_loss, accuracy, epoch_end-epoch_start))

        return self.model, running_loss, accuracy

    def test_epoch(self, dataloader, print_every=1):
        return self.train_epoch(dataloader, optimizer=None, mode='test')

    def get_accuracy_fns(self):
        '''
        Generate list of accuracy functions
        (not sure if necessary)
        '''
        accuracy_fns = {}

        def top1_corrects(outputs, label):
            # measure top 1 exact accuracy
            # assumes 1 label per data only
            _, pred  = outputs.topk(1)
            corrects = (pred == label).sum().item()
            return corrects
        accuracy_fns['top1'] = top1_corrects

        def approved_by_three(outputs, label):
            # implementation by paper
            # labels is assumes to be a 2d tensor
            _, pred = outputs.topk(1)
            correct_tensor = (pred == label).sum(dim=1).clamp(max=3) / 3
            corrects = correct_tensor.sum().item()
            return corrects
        accuracy_fns['approve3'] = approved_by_three

        def approve_by_3tuple(outputs, label):
            # approved_by_three but takes tuple of tensor as label
            _, pred = outputs.topk(1)
            corrects = 0
            for i in range(len(label)):
                pred_answer = pred[i]
                label_answer = label[i]
                correct_tensor = (pred_answer == label_answer)
                corrects += max(3, correct_tensor.sum().item())
            return corrects / 3
        accuracy_fns['approve3t'] = approve_by_3tuple
        return accuracy_fns
            

    # utility
    def print_every(self, iterr, total, every):
        if iterr % every == 0:
            print('    ....iteration {}/{}'.format(iterr, total), end='\r')

    def set_accuracy_fn(self, fn_code):
        if fn_code in self.get_accuracy_fns():
            self.accuracy_fn = self.get_accuracy_fns()[fn_code]

    def plot_over_epoch(self):
        '''
        Plot side-by-side the accuracy & losses stored
        in current statistics.
        '''
        if len(self.statistics.items()) < 4:
            raise ValueError("No training session has been run on this trainer")

        plt.figure()
        plt.subplot(121)
        plt.plot(self.statistics['train-losses'], color='purple')
        plt.plot(self.statistics['val-losses'], color='red')
        plt.title("Losses")

        plt.subplot(122)
        plt.plot(self.statistics['train-accuracy'], color='blue')
        plt.plot(self.statistics['val-accuracy'], color='cyan')

def main():
    print("DO NOT RUN THIS SCRIPT BY ITSELF\nIMPORT IT INSTEAD")

if __name__ == '__main__':
    main()