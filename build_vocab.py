import os
import torch
import collections

class DataHelper(object):
    def __init__(self, dataset, vocab=None):
       
        
        self.dataset = dataset

        self.base = '/content'

        self.current_set = os.path.join(self.base, 'cs-raw.txt')

        content, _ = self.get_content()

        self.build_vocab(content, min_count=20)


    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]
       

        label, content = zip(*content)

        return content, label


   

    def build_vocab(self, content, min_count=15):
     #   vocab = []
        a = collections.Counter(list(content[0].split()))
        for c in content[1:]:
            words = list(c.split(' '))
            a.update(words)

        
        result = dict(a)
        results = []
        for k,v in result.items():
          if v >= 5:
            results.append(k)
        
        results.insert(0, 'UNK')
        with open(os.path.join(self.base, 'vocab.txt'), 'w') as f:
            f.write('\n'.join(results))

        self.vocab = results





if __name__ == '__main__':
    data_helper = DataHelper(dataset='journal')
   # content, label = data_helper.get_content()
   # data_helper.build_vocab(content)
