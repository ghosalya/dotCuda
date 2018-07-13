import nltk
import pickle
import argparse
from collections import Counter
from vqaTools.vqa import VQA

class Answer(object):
    """Simple answers wrapper."""
    def __init__(self):
        self.ans2idx = {}
        self.idx2ans = {}
        self.idx = 0

    def add_ans(self, ans):
        if not ans in self.ans2idx:
            self.ans2idx[ans] = self.idx
            self.idx2ans[self.idx] = ans
            self.idx += 1

    def __call__(self, ans):
        if not ans in self.ans2idx:
            return self.ans2idx['<unk>']
        return self.ans2idx[ans]

    def __len__(self):
        return len(self.ans2idx)
    
def build_answers(annfile, quesfile):
    """Build an answers wrapper."""
    vqa = VQA(annfile, quesfile)
    counter = Counter()
    
    print('len of annotations dict:', len(vqa.dataset['annotations']))
    
    for ann_id in range(len(vqa.dataset['annotations'])): 
        ans_dict = vqa.dataset['annotations'][ann_id]['answers']

        for dic in ans_dict:
            counter[dic['answer']] += 1

        if (ann_id+1) % len(vqa.dataset['annotations']) == 0:
            print("[{}/{}] Answers tally completed.".format(ann_id+1, len(vqa.dataset['annotations'])))

    # print('counter', counter)
    # print('most common', counter.most_common(2))
    answers = [ans[0] for ans in counter.most_common(3000)]
    
    # Create an answer wrapper
    answer = Answer()

    # Add the words to the vocabulary.
    for i, ans in enumerate(answers):
        answer.add_ans(ans)
        
    return answer