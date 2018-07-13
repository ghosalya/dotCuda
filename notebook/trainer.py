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
    def __init__(self, model, device, accuracy_fn='approve3ts'):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        accuracy_fn_list = self.get_accuracy_fns()
        self.accuracy_fn = accuracy_fn_list[accuracy_fn]
        self.statistics = {}

    def train(self, train_dataset, val_dataset, epoch=5, batch_size=128, learnrate=1e-3, 
              collate_fn=None, save_every=1:
        '''
        Train over many epoch, outputing test result
        in between
        '''
        adam_optimizer = torch.optim.Adam(self.model.parameters(), learnrate)

        if collate_fn:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.statistics = {'train-losses':[],
                           'val-losses': [],
                           'train-accuracy': [],
                           'val-accuracy': [],
                           'train-losslist': [],
                           'train-acclist': [],
                           'val-losslist': [],
                           'val-acclist': []}

        for e in range(epoch):
            # training phase
            print('Epoch {} of {}'.format(e, epoch))
            print('  Training...')
            self.model, train_loss, train_acc, train_losslist, train_acclist = self.train_epoch(train_loader, optimizer=adam_optimizer)
            self.statistics['train-losses'].append(train_loss)
            self.statistics['train-accuracy'].append(train_acc)
            self.statistics['train-losslist'].append(train_losslist)
            self.statistics['train-acclist'].append(train_acclist)
            # validation phase
            print('  Validating...')
            self.model, val_loss, val_acc, val_losslist, val_acclist = self.test_epoch(val_loader)
            self.statistics['val-losses'].append(val_loss)
            self.statistics['val-accuracy'].append(val_acc)
            self.statistics['val-losslist'].append(val_losslist)
            self.statistics['val-acclist'].append(val_acclist)

            if e % save_every == 0:
                torch.save(self.model.state_dict(), './model_epoch{}.pt'.format(e))

        return self.model, self.statistics

    def train_epoch(self, dataloader, optimizer=None, mode='train', print_every=1, plot_every=5):
        '''
        Train function fro one epoch, returning the model, loss & accuracy
        '''
        if mode == 'train' and optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)

        self.model.train(mode == 'train')
        self.model.zero_grad()

        running_loss = 0
        running_correct = 0

        loss_list = []
        correct_list = []

        epoch_start = time.clock()
        iterr = 0
        for data in dataloader:
            iterr += 1

            # wrapping data
            question = data[0].long().cuda()
            image = data[1].to(self.device)
            labels = data[2].to(self.device)

            # forward pass
            self.model.zero_grad()
            outputs = self.model(image, question)

            # handling loss
            batch_loss = self.get_losses(outputs, labels)
            running_loss += batch_loss.item()

            # accuracy prediction
            running_correct += self.accuracy_fn(outputs, labels) / len(dataloader)

            # backpropagation
            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            self.print_every(iterr, len(dataloader), print_every, 
                             running_loss, running_correct)

            if iterr % plot_every == 0:
                loss_list.append(running_loss)
                correct_list.append(running_correct)
        
        epoch_end = time.clock()
        accuracy = running_correct 

        print('   >> Epoch finished with loss {:.5f} and accuracy {:.3f} in {:.4f}s'\
              .format(running_loss, accuracy, epoch_end-epoch_start))

        return self.model, running_loss, accuracy, loss_list, correct_list

    def test_epoch(self, dataloader, print_every=1):
        return self.train_epoch(dataloader, optimizer=None, mode='test')
    
    def get_losses(self, outputs, labels):
        '''
        This function expects labels to be a tensor of batchX10
        '''
        total_loss = None
        for i in range(10):
            if total_loss is None:
                total_loss = self.criterion(outputs, labels[:,i].long())
            else:
                total_loss += self.criterion(outputs, labels[:,i].long())
        return total_loss / (10*len(labels))

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
                if list(label_answer.size())[0] < 3:
                    continue
                #print(pred_answer, label_answer)
                label_size = list(label_answer.size())[0]
                for j in range(label_size):
                    a_idx = [k for k in range(label_size) if k != j]
                    masked_label = label_answer[a_idx]
                    correct_tensor = (pred_answer.long() == masked_label.long())
                    corrects += min(3, correct_tensor.sum().item()) / (3 * label_size * len(label))
            return corrects
        accuracy_fns['approve3t'] = approve_by_3tuple
        
        
        def approve_by_3tensor(outputs, label):
            # approved_by_three but takes tuple of tensor as label
            _, pred = outputs.topk(1)
            corrects = 0
            for i in range(10):
                sub_idx = [k for k in range(10) if k != i]
                label_subset = label[:, sub_idx]
                correct_answers  = (pred.long() == label_subset.long())
                corrects += correct_answers.sum(dim=1).clamp(max=3).sum() / (3 * label.size(0) * 10)
            return corrects
        accuracy_fns['approve3ts'] = approve_by_3tensor
        return accuracy_fns
            

    # utility
    def print_every(self, iterr, total, every, loss, acc):
        if iterr % every == 0:
            print('    ....iteration {}/{}   [loss {} || acc {}]'.format(iterr, total, loss, acc), end='\r')

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
        plt.subplot(1, 2, 1)
        plt.plot(self.statistics['train-losses'], color='purple')
        plt.plot(self.statistics['val-losses'], color='red')
        plt.title("Losses")

        plt.subplot(1, 2, 2)
        plt.plot(self.statistics['train-accuracy'], color='blue')
        plt.plot(self.statistics['val-accuracy'], color='cyan')
        plt.title("Accuracy")

        # plot in-epoch statistics
        nrow = len(self.statistics['train-losslist'])
        for i in range(nrow):
            plt.figure()
            # one row per epoch
            plt.subplot(1, 2, 1)
            plt.plot(self.statistics['train-losslist'][i], color='purple')
            plt.plot(self.statistics['val-losslist'][i], color='red')
            plt.title("Epoch {} - Losses".format(i))
            plt.subplot(1, 2, 2)
            plt.plot(self.statistics['train-acclist'][i], color='blue')
            plt.plot(self.statistics['val-acclist'][i], color='cyan')
            plt.title("Epoch {} - Accuracy".format(i))

    #### LEGACY METHODS ####
    def get_losses_legacy(self, outputs, labels):
        total_loss = None
        for i, label in enumerate(labels): # loop by batch
            one_data_loss = None
            if list(label.size())[0] < 3:
                continue
            for j in range(list(label.size())[0]):
                if one_data_loss is None:
                    one_data_loss = self.criterion(outputs[i:i+1].float(), label[j:j+1].long())
                else:
                    one_data_loss += self.criterion(outputs[i:i+1].float(), label[j:j+1].long())
            one_data_loss = one_data_loss / list(label.size())[0]
            # assigning to total loss
            if total_loss is None:
                total_loss = one_data_loss
            else:
                total_loss += one_data_loss
        return total_loss


def main():
    print("DO NOT RUN THIS SCRIPT BY ITSELF\nIMPORT IT INSTEAD")

if __name__ == '__main__':
    main()