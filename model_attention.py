import dgl
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import gensim

class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 vocab,
                 n_gram,
                 drop_out,
                 edges_num,
                 edges_matrix,
                 max_length=300,
                 trainable_edges=True,
                 pmi=None,
                 cuda=True,
          
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.vocab = vocab
     
        print(edges_num)


        self.node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)
        self.node_eta = torch.nn.Embedding.from_pretrained(torch.rand(len(vocab), 1), freeze=False)
       # self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)
       
        self.edges_num = edges_num
      
      #  self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.rand(edges_num, 1), freeze=False)

        self.hidden_size_node = hidden_size_node

        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('/content/glove.6B.300d.txt')))
        self.node_hidden.weight.requires_grad = True

        self.len_vocab = len(vocab)

        self.ngram = n_gram

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.max_length = max_length

        self.edges_matrix = edges_matrix

        self.dropout = torch.nn.Dropout(p=drop_out,inplace=True)

        self.activation = torch.nn.LeakyReLU(inplace=True)
        
        self.attn_fc = torch.nn.Linear(2 * hidden_size_node, 1, bias=True)

        self.Linear = torch.nn.Linear(hidden_size_node, class_num, bias=True)

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def load_word2vec(self, word2vec_file):
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        embedding_matrix = []

        for word in self.vocab:
          
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                # print(word)
            
                embedding_matrix.append(np.random.uniform(-0.1,0.1,300))

        embedding_matrix = np.array(embedding_matrix)

        return embedding_matrix


    def add_seq_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                try :
                 old_edge_id.append(self.edges_matrix[(src_word_old, dst_word_old)])
                except KeyError:
                 old_edge_id.append(np.random.randint(0,self.edges_num))
            # self circle
           # edges.append([src, src])
           # old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        local_vocab = set(doc_ids)

        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph()

        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)

        sub_graph.ndata['h'] = local_node_hidden
        sub_graph.ndata['eta'] = self.node_eta(local_vocab)
        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new)

        edges, old_edge_id = [], []
        # edges = []

        edges.extend(seq_edges)

        old_edge_id.extend(seq_old_edges_id)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)

        srcs, dsts = zip(*edges)
        sub_graph.add_edges(srcs, dsts)
        '''
        try:
            seq_edges_w = self.seq_edge_w(old_edge_id)
        except RuntimeError:
            print(old_edge_id)
        sub_graph.edata['w'] = seq_edges_w
        '''
        return sub_graph
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'w': F.leaky_relu(a)}

    def gcn_msg(self,edge):
      return {'m': edge.src['h'], 'w': edge.data['w']}

        
    def gcn_reduce(self,node):
      w = F.softmax(node.mailbox['w'], dim=1)
     # new_hidden = torch.sum(w * node.mailbox['m'], dim=1)
     # w = node.mailbox['w']
      new_hidden = torch.mul(w, node.mailbox['m'])
      new_hidden,_ = torch.max(new_hidden, 1)
      node_eta = node.data['eta']
      new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
      return {'h': new_hidden}  


    def forward(self, doc_ids):
        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]

        batch_graph = dgl.batch(sub_graphs)
        batch_graph.apply_edges(self.edge_attention)
        batch_graph.update_all(
           self.gcn_msg,self.gcn_reduce
        )

        h1 = dgl.sum_nodes(batch_graph, feat='h')

        drop1 = self.dropout(h1)
        act1 = self.activation(drop1)

        l = self.Linear(act1)
       
        return l
