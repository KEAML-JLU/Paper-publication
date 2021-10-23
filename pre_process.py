import random,re, os, nltk
from nltk import word_tokenize


def clean(text):
    
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"[\[\]\'{}(),.?!:_`]", " ", text)
    return text.lower()
   


def remove_short(text):
    results = []
    for word in text:
        if len(word) < 3:
            continue
        else:
            results.append(word)

    return results



class Ohsumed(object):
    def __init__(self):
        self.base = '/content'
      
  

    def make_set(self):
        set = 'data_cs'
        current = os.path.join(self.base, set)
        train_result = []
        val_result = []
        test_result = []
        result = []
        

        for dir, _, file_names in os.walk(current):
            length = len(file_names)
            if length == 0:
                continue
            train_index = int(0.8 * length)
            val_index = int(0.9 * length)
            train_file = file_names[:train_index] 
            val_file = file_names[train_index:val_index]
            test_file = file_names[val_index:]
            type = dir[17:19]
            for file in file_names:
              with open(os.path.join(dir,file)) as f:
                text = f.read()
                text = self.clean_text(text)
                result.append('\t'.join([type, text]))
            for file in train_file:
              with open(os.path.join(dir, file)) as f:
                text = f.read()
                text = self.clean_text(text)
                train_result.append('\t'.join([type, text]))

            for file in val_file:
              with open(os.path.join(dir, file)) as f:
                text = f.read()
                text = self.clean_text(text)
                val_result.append('\t'.join([type, text]))
            
            for file in test_file:
              with open(os.path.join(dir, file)) as f:
                text = f.read()
                text = self.clean_text(text)
                test_result.append('\t'.join([type, text]))
        with open('cs-raw.txt','w') as f:
            f.write('\n'.join(result))

        random.shuffle(train_result)
        with open('cs-train.txt','w') as f:
            f.write('\n'.join(train_result))
        

        random.shuffle(val_result)
        with open('cs-dev.txt','w') as f:
            f.write('\n'.join(val_result))


        random.shuffle(test_result)
        with open('cs-test.txt','w') as f:
            f.write('\n'.join(test_result))

    @staticmethod
    def clean_text(text):
     
  
        text = clean(text)
     
        text = word_tokenize(text) 
     
        
        return ' '.join(text)



if __name__ == '__main__':

   # nltk.download('stopwords')
    nltk.download('punkt')
   # nltk.download('averaged_perceptron_tagger')
   # nltk.download('wordnet')
    # stem_corpus()
    # shuffle_20ng()
    o = Ohsumed()
    o.make_set()
  