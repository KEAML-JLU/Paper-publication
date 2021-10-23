import torch
from model_attention import Model
from data_helper_multilabel_self_training import DataHelper
import numpy as np
import tqdm
import sys, random
import argparse
import time, datetime
import os
#from sklearn.metrics import f1_score


NUM_ITER_EVAL = 640
EARLY_STOP_EPOCH = 5
Train_batch = 64


def edges_mapping(vocab_len, content, ngram):
  
    d = {}
    count = 1
    zero = 0
  
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram)):
                dst = doc[dst_id]
  
                if count == 1:
                  d[(src, dst)] = count
                  count += 1
                else :
                  if (src, dst) not in d.keys():
                    d[(src, dst)] = count
                    count += 1
                    if src == 0 or dst == 0:
                      zero += 1

    print('zero : %d, poration: %.2f'%(zero,zero/count))
    return count, d


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset, dev_data_helper):
    model.eval()
    total_pred = 0
    correct = 0
    #iter = 0
    with torch.no_grad():
      for content, label, _ in dev_data_helper.batch_iter(batch_size=128, num_epoch=1):
          #iter += 1

          logits = model(content)
          #logits = torch.sigmoid(logits)
          #pred = torch.round(logits)
          pred = torch.argmax(logits, dim=1)
          #print(pred)
          #print(label)
          correct_pred = torch.sum(pred == label)
          correct += correct_pred
          #mi_f1 = f1_score(label.data.cpu().numpy(), pred.data.cpu().numpy(), average='micro')
          #correct += mi_f1
          total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    return torch.div(correct, total_pred)


def test(model, dataset):
    model.eval()
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
            _,pred = torch.sort(logits, dim=1, descending=True)
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
    '''
    correct = 0
    iter = 0
    with torch.no_grad():
      for content, label, _ in data_helper.batch_iter(batch_size=128, num_epoch=1):
          iter += 1
          logits = model(content)
          logits = torch.sigmoid(logits)
          pred = torch.round(logits)
          mi_f1 = f1_score(label.data.cpu().numpy(), pred.data.cpu().numpy(), average='micro')
          correct += mi_f1

    #correct = correct.float()
    return correct / iter
    '''

def prepare_self_train(model, idx, update_interval, data_helper):
  target_num = min( Train_batch * update_interval, len(data_helper.content))
  if idx + target_num >= len(data_helper.content):
    select_idx = torch.cat((torch.arange(idx, len(data_helper.content)),
                torch.arange(idx + target_num - len(data_helper.content))))
  else:
    select_idx = torch.arange(idx, idx + target_num)
  assert len(select_idx) == target_num

  idx = (idx + len(select_idx)) % len(data_helper.content)
  #select_dataset = data_helper.content[select_idx] # may error
  #select_label = data_helper.label[select_idx]

  all_preds, all_input_ids = inference(model, data_helper, select_idx)
  weight = all_preds**2 / torch.sum(all_preds, dim=0)
  target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
  all_target_pred = target_dist.argmax(dim=-1)
  agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
  self_train_dict = {'input_ids':all_input_ids, 'labels':target_dist}

  return self_train_dict, idx, agree

def inference(model, data_helper, select_idx):
  model.eval()
  all_preds = []
  all_input_ids = []
  try :
    for content in data_helper.target(select_idx, batch_size=128):
      with torch.no_grad():
        logits = model(content)
        logits = torch.sigmoid(logits)
        all_preds.append(logits)
        
        all_input_ids += content
    all_preds = torch.cat(all_preds, dim=0)
    #all_input_ids = torch.cat(torch.tensor(all_input_ids), dim=0)
    return all_preds, all_input_ids
  except RuntimeError as err:
    print("error!")


def train(ngram, name, bar, drop_out, dataset, is_cuda, edges=True):
    print('load data helper.')
    with open('/content/data/data_cs/vocab.txt','r') as f:
      vocab = f.read()
      vocab = vocab.split('\n')
    self_training_epoch = 5
    early_stop = True
    data_helper = DataHelper(dataset=dataset, mode='train', vocab=vocab)
    count, edges_mappings = edges_mapping(len(data_helper.vocab), data_helper.content, ngram)
    model = Model(class_num=len(data_helper.labels_str), hidden_size_node=300,
                      vocab=data_helper.vocab, n_gram=ngram, drop_out=drop_out, edges_matrix=edges_mappings, edges_num=count,
                      trainable_edges=edges, pmi=None, cuda=is_cuda)
    if os.path.exists(os.path.join('.', 'model_journal'+'.pth')) :
        print('load model from file.')
    
        model.load_state_dict(torch.load(os.path.join('.', 'model_journal' + '.pth')))

        return model
    else:
        print('new model.')
       
        print('count finished! next is run model')
       
    
    dev_data_helper = DataHelper(dataset=dataset, mode='dev', vocab=vocab)
    print(model)

    if is_cuda:
        print('cuda')
        model.cuda()
    total_step = int(len(data_helper.content) * self_training_epoch / Train_batch)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')#torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)
    idx = 0
    update_interval = 100
    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best = 0.0
    last_best_epoch = 0
    start_time = time.time()
    #total_loss = 0.0
    #total_correct = 0
    #total = 0
    if early_stop:
      agree_count = 0 

    for i in range(int(total_step / update_interval)): #57
      
      self_train_dict, idx, agree = prepare_self_train(model, idx, update_interval, data_helper)
      #all_preds = inference(model, dataset, data_helper)
      #weight = all_preds**2 / torch.sum(all_preds, dim=0)
      #target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
      #all_target_pred = target_dist.argmax(dim=-1)
      #agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
      if early_stop:
        if 1 - agree < 1e-3:
          agree_count += 1
        else:
          agree_count = 0
        if agree_count > 3:
          break
      
      model.train()
      total_train_loss = 0
      model.zero_grad()
      #target_dist = self_train_dict['labels']
      #print(target_dist)
    
      for content, label in data_helper.batch_iter_target(batch_size=Train_batch, train_dict=self_train_dict):
        improved = ''
        logits = model(content)
        preds = torch.nn.LogSigmoid()(logits)
        loss = loss_func(preds.view(-1, len(data_helper.labels_str)), label.view(-1, len(data_helper.labels_str)))
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.zero_grad()
      
      #avg_train_loss = torch.tensor([total_train_loss / self_train_dict['labels'].shape[0]])
        '''
        if epoch < 2 :
          model.eval()
          logits = model(content)
          logits = torch.sigmoid(logits)
          fake_pred = torch.round(logits)
          model.train()
          optim.zero_grad()
          logits = model(content)
          loss = ((T1-epoch) / (T2-T1)) * af * loss_func(logits, fake_pred.float())
          #mi_f1 = f1_score(label.data.cpu().numpy(), fake_pred.data.cpu().numpy(), average='micro')
        
        else:
          logits = model(content)
          #logits = torch.sigmoid(logits)
          #pre = torch.round(logits)
          loss = loss_func(logits, label.float())
          optim.zero_grad()
          #mi_f1 = f1_score(label.data.cpu().numpy(), pre.data.cpu().numpy(), average='micro')
        
        if epoch > 1:
          model.eval()
          logits = model(content)
          fake_pred = torch.argmax(logits, dim=1)
          model.train()
          optim.zero_grad()
          logits = model(content)
          loss = loss_func(logits, fake_pred)
          correct = torch.sum(fake_pred == label)

        else:
     
          logits = model(content)
          loss = loss_func(logits, label)
          pred = torch.argmax(logits, dim=1)
          optim.zero_grad()
          correct = torch.sum(pred == label)
        

        total += 1

        total_loss += loss.item()

        #optim.zero_grad()
        loss.backward()
        optim.step()
        '''
        iter += 1
        if bar:
            pbar.update()
        if iter % NUM_ITER_EVAL == 0:
            if bar:
                pbar.close()
            val = dev(model, dataset, dev_data_helper)
            if val > best:
                best = val
                #last_best_epoch = epoch
                improved = '*'
                torch.save(model.state_dict(), name + '.pth')

            #if epoch - last_best_epoch >= EARLY_STOP_EPOCH:
                #return model
            msg = 'Updata_interval: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Micro_score: {6:>7.2%}' \
                  + 'Val acc: {2:>7.2%}, Time: {3}{4}' \
                  # + ' Time: {5} {6}'

            print(msg.format(i, iter, val, get_time_dif(start_time), improved, total_train_loss/ NUM_ITER_EVAL,
                             #,float(total_correct) / float(NUM_ITER_EVAL)
                             0))

            total_train_loss = 0.0
            #total_correct = 0
            #total = 0
            if bar:
                pbar = tqdm.tqdm(total=NUM_ITER_EVAL)

    return model


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
    result, _, _ = test(model, args.dataset)

    
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