import torch
from model_attention import Model
from data_helper import DataHelper
import numpy as np
import tqdm
import sys, random
import argparse
import time, datetime
import os
NUM_ITER_EVAL = 500
EARLY_STOP_EPOCH = 3

def precision_recall_fscore_k(y_true,y_pred,num):
    if not isinstance(y_pred[0],list):
        y_pred=[[each] for each in y_pred]
#     print(y_pred)
#    y_pred=[each[0:num] for each in y_pred]
#    unique_label=count_unique_label(y_true,y_pred)
    unique_label = [i for i in range(100)]
    #计算每个类别的precision、recall、f1-score、support
    res={}
    result=''
    tps = 0
    tp_fps = 0
    tp_fns = 0
    for each in unique_label:
        cur_res=[]
        tp_fn=y_true.count(each)#TP+FN
        tp_fns += tp_fn
        #TP+FP
        tp_fp=0
        for i in y_pred:
            if each in i:
                tp_fp+=1
        tp_fps += tp_fp
        #TP
        tp=0
        for i in range(len(y_true)):
            if y_true[i] == each and each in y_pred[i]:
                tp+=1
        tps += tp
      #  support=tp_fn
        try:
            precision=round(tp/tp_fp,2)
            recall=round(tp/tp_fn,2)
      #      f1_score=round(2/((1/precision)+(1/recall)),2)
        except ZeroDivisionError:
            precision=0
            recall=0
      #      f1_score=0
        cur_res.append(precision)
        cur_res.append(recall)
       # cur_res.append(support)
        res[str(each)]=cur_res
    #title='\t'+'precision@'+str(num)+'\t'+'recall@'+str(num)+'\t'+'f1_score@'+str(num)+'\t'+'support'+'\n'
   # result+=title

    sums=len(unique_label)
    precision_macro = 0
    recall_macro = 0

    for k,v in res.items():
      precision_macro += v[0]
      recall_macro += v[1]
    
    precision_macro = precision_macro / sums
    recall_macro = recall_macro / sums
    precision_micro = tps / tp_fps
    recall_micro = tps / tp_fns
    f1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro)
    f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)

  
    result = (result+'top-%s'%num+'\t'+str(round(precision_macro,2))+'\t'+str(round(recall_macro,2))+'\t'+str(round(f1_macro,2))+'\t'
    +str(round(precision_micro,2))+'\t'+str(round(recall_micro,2))+'\t'+str(round(f1_micro,2)))
    return result
#统计所有的类别
def count_unique_label(y_true,y_pred):
    unique_label=[]
    for each in y_true:
        if each not in unique_label:
            unique_label.append(each)
    for i in y_pred:
        for j in i:
            if j not in unique_label:
                unique_label.append(j)
    unique_label=list(set(unique_label))
    return unique_label


def edges_mapping(vocab_len, content, ngram):
   # row = []
   # col = []
   # add = []
   # data = []
    d = {}
    count = 1
    zero = 0
   # mapping = np.zeros(shape=(vocab_len, vocab_len), dtype=np.int32)
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram)):
                dst = doc[dst_id]
               # if [src,dst] not in add:
               #   add.append([src,dst])
               #   data.append(count)
               #   count += 1
               #   row.append(src)
               #   col.append(dst)
                '''
                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                '''
                if count == 1:
                  d[(src, dst)] = count
                  count += 1
                else :
                  if (src, dst) not in d.keys():
                    d[(src, dst)] = count
                    count += 1
                    if src == 0 or dst == 0:
                      zero += 1
    #row, col = zip(*add)
   # mapping = csr_matrix((data, (row,col)))
   # for word in range(vocab_len):
   #     mapping[word, word] = count
   #     count += 1
   # del mapping
   # gc.collect()
    print('zero : %d, poration: %.2f'%(zero,zero/count))
    return count, d


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset, dev_data_helper):
    model.eval()
    #data_helper = DataHelper(dataset, mode='dev')
    total_pred = 0
    correct = 0
    iter = 0
    with torch.no_grad():
      for content, label, _ in dev_data_helper.batch_iter(batch_size=128, num_epoch=1):
          iter += 1

          logits = model(content)
          pred = torch.argmax(logits, dim=1)

          correct_pred = torch.sum(pred == label)

          correct += correct_pred
          total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    # print(torch.div(correct, total_pred))
    return torch.div(correct, total_pred)


def test(model, dataset):
    model.cuda()
    data_helper = DataHelper(dataset=dataset, mode='test')
    topk = (1,3,5,10)
    result = []
    tests_pred = []
    tests_label = []
    for k in topk:
      total_pred = 0
      correct = 0
      iter = 0
      test_pred = []
      test_label = []
      model.eval()
      with torch.no_grad():
        for content, label, _ in data_helper.batch_iter(batch_size=128, num_epoch=1):
            iter += 1

            labels = label.cpu().numpy().tolist()
            test_label.extend(labels)

            logits = model(content)
            _,pred = torch.sort(logits, dim=1,descending=True)
            temp_pred = pred[: ,:k]
           
            label = label.view(-1,1)
            correct_pred = torch.sum(temp_pred == label)

            correct += correct_pred
            total_pred += len(content)

          
            temp_preds = temp_pred.cpu().numpy().tolist()
            
            
            test_pred.extend(temp_preds)
        
        tests_label.append(test_label)
        tests_pred.append(test_pred)

        total_pred = float(total_pred)
        correct = correct.float()
        result.append(torch.div(correct, total_pred).to('cpu').numpy())
    return result, tests_label, tests_pred


def train(ngram, name, bar, drop_out, dataset, is_cuda, edges=True):
    print('load data helper.')
    with open('/content/data/data_cs/vocab.txt','r') as f:
      vocab = f.read()
      vocab = vocab.split('\n')
    data_helper = DataHelper(dataset=dataset, mode='train', vocab=vocab)
    count, edges_mappings = edges_mapping(len(data_helper.vocab), data_helper.content, ngram)
    model = Model(class_num=len(data_helper.labels_str), hidden_size_node=300,
                      vocab=data_helper.vocab, n_gram=ngram, drop_out=drop_out, edges_matrix=edges_mappings, edges_num=count,
                      trainable_edges=edges, pmi=None, cuda=is_cuda)
    if os.path.exists(os.path.join('.', 'temp_model_journal'+'.pth')) :
        print('load model from file.')
    
        model.load_state_dict(torch.load(os.path.join('.', 'temp_model_journal' + '.pth')))

        return model
    else:
        print('new model.')
       
        print('count finished! next is run model')
       
    
    dev_data_helper = DataHelper(dataset=dataset, mode='dev', vocab=vocab)
    print(model)
    if is_cuda:
        print('cuda')
        model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)

    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_acc = 0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for content, label, epoch in data_helper.batch_iter(batch_size=128, num_epoch=100):
        improved = ''
        model.train()

        logits = model(content)
        loss = loss_func(logits, label)

        pred = torch.argmax(logits, dim=1)

        correct = torch.sum(pred == label)

        total_correct += correct
        total += len(label)

        total_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        iter += 1
        if bar:
            pbar.update()
        if iter % NUM_ITER_EVAL == 0:
            if bar:
                pbar.close()

            val_acc = dev(model, dataset, dev_data_helper)
            if val_acc > best_acc:
                best_acc = val_acc
                last_best_epoch = epoch
                improved = '*'

                torch.save(model.state_dict(), name + '.pth')

            if epoch - last_best_epoch >= EARLY_STOP_EPOCH:
                return model
            msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Acc: {6:>7.2%}' \
                  + 'Val Acc: {2:>7.2%}, Time: {3}{4}' \
                  # + ' Time: {5} {6}'

            print(msg.format(epoch, iter, val_acc, get_time_dif(start_time), improved, total_loss/ NUM_ITER_EVAL,
                             float(total_correct) / float(total)))

            total_loss = 0.0
            total_correct = 0
            total = 0
            if bar:
                pbar = tqdm.tqdm(total=NUM_ITER_EVAL)

    return model


def word_eval():
    print('load model from file.')
    data_helper = DataHelper('r8')
    edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, 1)
    model = torch.load(os.path.join('word_eval_1.pkl'))

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()

    core_word = 'billion'
    core_index = data_helper.vocab.index(core_word)

    results = {}
    for i in range(len(data_helper.vocab)):
        word = data_helper.vocab[i]
        n_word = edges_matrix[i, core_index]
        # n_word = edges_matrix[i, i]
        if n_word != 0:
            results[word] = edges_weights[n_word][0]

    sort_results = sorted(results.items(), key=lambda d: d[1])

    print(sort_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', required=False, type=int, default=4, help='ngram number')
    parser.add_argument('--name', required=False, type=str, default='model', help='project name')
    parser.add_argument('--bar', required=False, type=int, default=1, help='show bar')
    parser.add_argument('--dropout', required=False, type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--edges', required=False, type=int, default=1, help='trainable edges')
    parser.add_argument('--rand', required=False, type=int, default=7, help='rand_seed')

    args = parser.parse_args()

    print('ngram: %d' % args.ngram)
    print('project_name: %s' % args.name)
    print('dataset: %s' % args.dataset)
    print('trainable_edges: %s' % args.edges)
    # #
    SEED = args.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.bar == 1:
        bar = True
    else:
        bar = False
    
    if args.edges == 1:
        edges = True
        print('trainable edges')
    else:
        edges = False

    model = train(args.ngram, args.name, bar, args.dropout, dataset=args.dataset, is_cuda=True, edges=edges)
    model.load_state_dict(torch.load(os.path.join('.', 'model' + '.pth')))
    result, tests_label, tests_pred = test(model, args.dataset)
    print('top-1 test acc: ', result[0])
    print('top-3 test acc: ', result[1])
    print('top-5 test acc: ', result[2])
    print('top-10 test acc: ', result[3])
    '''
    for i, num in enumerate([1,3,5,10]):

     print(precision_recall_fscore_k(tests_label[i], tests_pred[i], num))
    
    for i, idx in enumerate([1,3,5,10]):
      print("top-%d Macro average Test Precision, Recall and F1-Score..."%idx)
      print(metrics.precision_recall_fscore_support(tests_label[i], tests_pred[i], average='macro'))
    for i, idx in enumerate([1,3,5,10]): 
      print("top-%d Micro average Test Precision, Recall and F1-Score..."%idx)
      print(metrics.precision_recall_fscore_support(tests_label[i], tests_pred[i], average='micro'))
    '''
