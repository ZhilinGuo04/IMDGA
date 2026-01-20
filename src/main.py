import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import copy
import math
import time
import argparse
import shap
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import string
import pickle
import re
from gnn import GCN
from utils import *
from logger import *
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.utils import degree
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import DebertaTokenizer, DebertaForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from tag_pipeline import TAG_Pipeline
from transformers import pipeline
from transformers import AutoModel
from gnnshap.explainer import GNNShapExplainer
from sklearn.metrics import f1_score, accuracy_score,recall_score
from data_utils import tag_dataset 

filter_words = [' ', ',', 'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']


class Graph_Text(object):
    def __init__(self, text, neighbor, weights, label):
        self.label = label
        self.pre_label = 0
        self.text = text
        self.orig_text = 0
        self.words = 0
        self.neighbor = neighbor
        self.attack_words = 0
        self.query = 0
        self.success = False
        self.attack_number = 0 
        self.sim = 0.0
        self.change = 0 
        self.mask = None 
        self.wrong_mask = None 
        self.weights = weights
        self.attack_nodes = []

class Score():
    def __init__(self, target_index, score_normal, score_important):
        self.target_index = 0
        self.score_normal = 0
        self.score_important = 0
        self.word = None

class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts

    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # ， DataLoader 
        return self.texts[idx]
def is_word(s):
    return bool(s and re.match(r'^[a-zA-Z]+$', s))


def node_neighbor(node_idx, edge_index, device, num_nodes , num_hops = 2, ):
    
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(edge_index.size(1), dtype=torch.bool)
    my_edge_mask = row.new_empty(row.size(0), dtype=torch.bool) # added by sakkas
    my_edge_mask.fill_(False)
    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=device).flatten()
    else:
        node_idx = node_idx.to(device)
    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)# input, dimension, index
        my_edge_mask[edge_mask] = True
        subsets.append(col[edge_mask])
    subset = torch.cat(subsets).unique()
    return subset

def text2emb(texts,device, batch_size = 32):
    model_name = config.get_llm_path('bert') 
    tokenizer = BertTokenizer.from_pretrained(model_name) 
    model = AutoModel.from_pretrained(model_name).to(device)
    all_embeds = []
    model.eval()
    for i in range(0, len(texts),batch_size):
        text = texts[i:i+batch_size]
        inputs = tokenizer(
            text,
            padding = 'max_length',
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}  
        with torch.no_grad():
            outputs = model(**inputs)
            bert_embeds = outputs.last_hidden_state 
            bert_embeds = bert_embeds.mean(1)
        all_embeds.append(bert_embeds)

    feat = torch.cat(all_embeds, dim=0)  
    return feat
    
def sentence2emb(sentence,tokenizer,model,device):
    inputs = tokenizer(
        sentence,
        padding = 'max_length',
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    outputs = model(**inputs)
    bert_embeds = outputs.last_hidden_state 
    bert_embeds = bert_embeds.mean(1)
    return bert_embeds

def train_model(model,data,args):
    save_path = f'./data/model/best_weights_{args.model}_{args.llm}_{args.num_layers}_{args.dataset}_{args.epochs}_{args.lr}_{args.train_ratio}_{args.val_ratio}_{args.test_ratio}.pt'
    if os.path.exists(save_path):
        best_weights = torch.load(save_path)
        model.load_state_dict(best_weights)
    else:
        final_train_acc, best_val, final_test = 0,0,0
        best_weights = None
        if args.epochs > 0:
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2decay)
        tot_time = 0
        for epoch in tqdm(range(1, args.epochs + 1)):
            start = time.time()
            loss = train(model, data.x, data.edge_index, data.y, data.train_indices, optimizer)
            if epoch > args.epochs / 2 and epoch % args.test_freq == 0 or epoch == args.epochs:
                test_f1, _, _  = test(model, data,data.test_indices)
                val_f1, _, _  = test(model, data,data.val_indices)
                val = val_f1
                tst = test_f1
                if val > best_val :
                    best_val = val
                    final_test = tst
                    if args.best_weights:
                        best_weights = deepcopy(model.state_dict())
                    print("Best Test Result: {}".format(test_f1))
            stop = time.time()
            tot_time += stop-start
        
        print(f'Avg train time {tot_time/args.epochs}')
        torch.save(model.state_dict(), save_path)

def train(model, x, edge_index, y, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, edge_index)
    # transductive setting
    if train_idx.size(0) < y.size(0):
        out = out[train_idx]
        y = y[train_idx]
    loss = F.nll_loss(out, y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()    


@torch.no_grad()
def test(model,data,idx,pred=None):
    """
    Evaluates the GNN model on node-level classification tasks using full-batch evaluation.

    Args:
        feature (torch.Tensor): The input features for the test nodes.
    
    Returns:
        tuple: A tuple containing:
            - F1_score (float): The micro-averaged F1 score on the test dataset.
            - Accuracy (float): The accuracy on the test dataset.
            - Recall (float): The micro-averaged recall on the test dataset.
    """
    model.eval()
    y_pred = torch.softmax(model(data.x, data.edge_index),-1).cpu()
    y_pred_test = y_pred[idx.cpu()]
    y = data.y.cpu()
    y_pred_test = torch.argmax(y_pred_test, axis=1)
    test_f1 = f1_score(y[idx.cpu()], y_pred_test, average="micro")
    test_acc = accuracy_score(y_true=y[idx.cpu()], y_pred=y_pred_test) 
    test_recall = recall_score(y_true=y[idx.cpu()], y_pred=y_pred_test, average="micro")
    
    return test_f1,test_acc,test_recall



def remove_edges(data_edge_index, graph, vul_edge_index):
    """
    Remove edges from data_edge_index that are in vul_edge_index (including bidirectional edges).
    
    Args:
        data_edge_index: Original edge index, shape [2, num_edges].
        graph: NetworkX graph object.
        vul_edge_index: Edges to remove, shape [2, num_vul_edges].
    
    Returns:
        Updated edge index after removing edges.
    """
    mask = torch.ones(data_edge_index.size(1), dtype=torch.bool, device=data_edge_index.device)
    
    for i in range(vul_edge_index.size(1)):
        edge = vul_edge_index[:, i]
        u, v = int(edge[0]), int(edge[1])
        
        if graph.has_edge(u, v) or (isinstance(graph, nx.Graph) and graph.has_edge(v, u)):
            edge_mask = (data_edge_index[0] == edge[0]) & (data_edge_index[1] == edge[1])
            mask[edge_mask] = False
            if isinstance(graph, nx.Graph):
                reverse_edge_mask = (data_edge_index[0] == edge[1]) & (data_edge_index[1] == edge[0])
                mask[reverse_edge_mask] = False
        else:
            print(f"Edge ({u}, {v}) not in graph, skipping")
    
    return data_edge_index[:, mask]


def find_attack_nodes_random(data,correct_mask,device):
    row, col, value = data.adj_t.coo()   
    data.edge_index = torch.stack([row, col], dim=0)
    degree_num =  degree(data.edge_index[0,:])
    indices = torch.arange(len(degree_num)).to(device)
    tmp_mask1 = torch.isin(indices, data.test_indices)
    tmp_mask2 = (data.text_len > 30) 
    mask_1 = tmp_mask1 & tmp_mask2
    mask = correct_mask & mask_1
    tmp_indices = mask.nonzero(as_tuple=True)[0]
    random_index = torch.randperm(mask.nonzero(as_tuple=True)[0].size(0))
    attack_idx=  torch.tensor(tmp_indices[random_index][:int(data.test_indices.size(0)*0.1)])
    return attack_idx


def find_neighbors_with_weights(graph, nodes,device):
    all_neighbors_with_weights = []

    for node in nodes:
        node = int(node)  
        
        current_node_neighbors = {} #  {node_idx: weight} 
        
        # 1. 
        current_node_neighbors[node] = 1.0 #  1.0
        
        try:
            degree_v = graph.degree(node)
        except nx.NetworkXError:
            # ，， 0。
            #  0 （），。
            # ，，。
            #  'nodes' 。
            print(f"Warning: Node {node} not found in graph. Skipping.")
            all_neighbors_with_weights.append(([], []))
            continue # ，
        
        if degree_v == 0:
            # ，
            all_neighbors_with_weights.append(([node], [1.0]))
            continue

        
        first_hop_neighbors = set(graph.neighbors(node))
        for neighbor_1 in first_hop_neighbors:
            weight_1_hop = 1.0 / math.sqrt(degree_v)
            current_node_neighbors[neighbor_1] = weight_1_hop

            
            try:
                degree_u = graph.degree(neighbor_1)
            except nx.NetworkXError:
                # ， neighbor_1 ，
                continue 
            
            if degree_u == 0: # ，
                continue

            for neighbor_2 in graph.neighbors(neighbor_1):
                # ，
                if neighbor_2 != node and neighbor_2 not in first_hop_neighbors:
                    weight_2_hop = (1.0 / math.sqrt(degree_v)) * (1.0 / math.sqrt(degree_u))
                    # ，，
                    # 。，。
                    if neighbor_2 not in current_node_neighbors:
                        current_node_neighbors[neighbor_2] = weight_2_hop


        
        # ，
        # sorted_neighbors_items = sorted(current_node_neighbors.items(), key=lambda item: item[0])
        
        neighbors_list_for_node = [item[0] for item in current_node_neighbors.items()]
        weights_list_for_node = [item[1] for item in current_node_neighbors.items() ]
        
        all_neighbors_with_weights.append((neighbors_list_for_node, weights_list_for_node))

    
    final_neighbor_list = [item[0] for item in all_neighbors_with_weights]
    final_weights_list = [torch.tensor(item[1]).to(device) for item in all_neighbors_with_weights]
    
    return final_neighbor_list, final_weights_list

def pagerank(graph, k):
    n = graph.number_of_nodes()
    M = np.zeros((n, n))
    for j in range(n):
        neighbors = list(graph.neighbors(j))
        degree = len(neighbors) 
        if degree > 0:
            for i in neighbors:
                M[i, j] = 1.0 / degree 
    
    Mk = np.linalg.matrix_power(M, k)

    return Mk




def _tokenize(seq, tokenizer):
    seq = seq.strip()
    seq = seq.replace('\n', '')

    # Use a simple regex or iterative replacement to separate punctuation
    # This loop ensures that punctuation like '.', ',', '!', '?' are
    # surrounded by spaces, effectively making them separate "words"
    for punct in string.punctuation:
        seq = seq.replace(punct, f' {punct} ')

    # Split the sequence by spaces, then filter out empty strings
    words = [word for word in seq.split(' ') if word]

    sub_words = []
    keys = []
    index = 0
    for i, word in enumerate(words):
        # Tokenize the individual word
        sub = tokenizer.tokenize(word)
        sub_words.extend(sub)
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def roberta_tokenize(seq, tokenizer):
    seq = seq.strip()
    seq = seq.replace('\n', '')

    # Use a simple regex or iterative replacement to separate punctuation
    # This loop ensures that punctuation like '.', ',', '!', '?' are
    # surrounded by spaces, effectively making them separate "words"
    tokenizer.add_prefix_space = True
    # for punct in string.punctuation.replace('-', ''):
    #     seq = seq.replace(punct, f' {punct} ')
    # Split the sequence by spaces, then filter out empty strings
    words = [word for word in seq.split(' ') if word]

    sub_words = []
    keys = []
    index = 0
    for i, word in enumerate(words):
        # Tokenize the individual word
        sub = tokenizer.tokenize(word)
        sub_words.extend(sub)
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def get_substitues(substitutes, tokenizer, substitutes_score=None, threshold=2.0):
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
            
    return words

def get_important_scores(attack_idx, data, graph_text, target_model,llm_emb_model, orig_prob, orig_label, tokenizer, max_length,device):
    masked_words = _get_masked(graph_text.words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []
    dataset = TextDataset(texts)
    Text_dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,  
        num_workers=0,  
    )
    for batch_text in Text_dataloader:
        inputs = tokenizer.batch_encode_plus(batch_text,
                                            add_special_tokens=True,
                                            max_length=max_length,
                                            padding='max_length',  # max_lengthpadding
                                            truncation=True,        # max_lengthtruncation
                                            return_tensors='pt' )
        input_ids, token_type_ids,attention_mask = inputs["input_ids"], inputs["token_type_ids"],inputs['attention_mask']
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.cat(all_input_ids, dim=0)
    masks = torch.cat(all_masks, dim=0)
    segs = torch.cat(all_segs, dim=0)
    seqs = seqs.to(device)
    

    bert_embeds = text2emb(texts,device=device)
    features = torch.tensor(bert_embeds).to(device)
    
    
    important_scores = []
    for feature in features:
        data.x[attack_idx] = feature
        logits = torch.softmax(target_model(data.x,data.edge_index)[graph_text.neighbor], -1)
        logits_diff = orig_prob - logits
        score = torch.gather(logits_diff, 1, orig_label.unsqueeze(1)).squeeze(1)
        score = score * graph_text.weights.to(device)
        score = torch.abs(score)
        score = float(score.sum())
        important_scores.append(score)
        
    return important_scores


def compute_normalized_margin_prob(logits,normalize = True):
    """
    Compute the difference between the top two probabilities from a logits vector.
    
    Args:
        logits (torch.Tensor): Input logits tensor of shape (n,)
        
    Returns:
        tuple: (diffs, max_diff)
            - diffs (torch.Tensor): Differences between top-1 and top-2 probabilities, shape (n,1)
            - max_diff (float): Maximum difference value
    """
    # Get top-2 probabilities
    top2_probs, _ = torch.topk(logits, k=2, dim=-1)
    
    # Compute difference between top-1 and top-2 probabilities
    diffs = top2_probs[:, 0:1] - top2_probs[:, 1:2]
    
    # Compute maximum difference
    max_diff = diffs.max().item()

    diffs = diffs.squeeze()

    if normalize:
        diffs = diffs / max_diff

    return diffs

def compute_normalized_similarity(emb, target_index, neighbor_indices):
    """
    Compute normalized cosine similarity between target embedding and neighbor embeddings.
    
    Args:
        emb (torch.Tensor): Embedding tensor of shape (n, d), where n is number of embeddings, d is embedding dimension
        target_index (int): Index of the target embedding
        neighbor_indices (list): List of indices for neighbor embeddings
    
    Returns:
        torch.Tensor: Normalized cosine similarities, shape (len(neighbors_index),)
    """
    # Extract target and neighbor embeddings
    target_emb = emb[target_index]  # Shape: (1, d)
    neighbors_emb = emb[neighbor_indices]  # Shape: (len(neighbor_indices), d)
    
    # Compute cosine similarity
    similarities = F.cosine_similarity(target_emb, neighbors_emb, dim=-1)  # Shape: (len(neighbor_indices),)
    
    # Normalize by maximum similarity
    max_sim = similarities.max()
    if max_sim > 0:  # Avoid division by zero
        normalized_similarities = similarities / max_sim
    else:
        normalized_similarities = similarities  # If max_sim is 0, return original similarities
    
    return normalized_similarities

def check_label_same(label, target_index, neighbor_indices):
    """
    Check if neighbor nodes have the same label as the target node.
    
    Args:
        label (torch.Tensor): Label tensor of shape (n,), where n is number of nodes
        target_index (int): Index of the target node
        neighbor_indices (list): List of indices for neighbor nodes
    
    Returns:
        list: List of 0/1 indicating if each neighbor's label matches the target label
    """
    # Get target label
    target_label = label[target_index]
    
    # Get neighbor labels
    neighbor_labels = label[neighbor_indices]
    
    # Compare neighbor labels with target label
    same_label = (neighbor_labels == target_label).squeeze().long()
    
    return same_label

def find_vulnerable_nodes(margin_prob, text_sim, label, assortativity, beta, neighbor_indices, top_k=5):
    top_k = min(len(neighbor_indices), top_k)
    # top_k = len(neighbor_indices)
    if assortativity < beta:
        vulnerability = (1 - margin_prob) + (label * beta) + (label * text_sim)
    else:
        vulnerability = (1 - margin_prob) + torch.relu(beta - label) + (1 - text_sim)   
    _ , topk_indices = torch.topk(vulnerability, top_k)
    topk_indices = topk_indices.tolist()
    neighbor_indices = np.array(neighbor_indices)
    return neighbor_indices[topk_indices]



def find_vulnerable_neighbor(target_node, node_neighbor, node_degree, logits, P, device,graph=None, w =[0.3, 0.6, 0.1],topk = 10):
    score = torch.zeros(len(node_neighbor)).to(device)

    feature_influence = torch.tensor(P[target_node.cpu()][node_neighbor.cpu()]).to(device)
    margin_prob = compute_normalized_margin_prob(logits)
    degree = node_degree[node_neighbor]
    score_i = w[0] * (1 - margin_prob) + w[1] * (feature_influence) + w[2] * (1.0/degree)
    score += score_i
    
    node_neighbor = torch.tensor(node_neighbor)
    topk = min(topk, node_neighbor.size(0))
    values, indices = torch.topk(score, k = int(topk))
    Vulnerable_neighbors = node_neighbor[indices]
    target = torch.tensor([target_node]).to(device)
    
    Vulnerable_neighbors = torch.cat((Vulnerable_neighbors, target))
    one_hop = torch.tensor(list(graph.neighbors(int(target_node)))).to(device)
    Vulnerable_neighbors = torch.unique(torch.cat((one_hop, Vulnerable_neighbors)))
    del feature_influence, one_hop
    return Vulnerable_neighbors
    
def attack(args, logger, attack_idx, data ,graph_text, target_model, llm_mask_model, llm_emb_model, tokenizer, top_k, batch_size, index, shap_values_pred, device, graph = None ,max_length=512,P = None):
    if args.llm == 'roberta' or args.llm == 'deberta':
        words, sub_words, keys = roberta_tokenize(graph_text.text, tokenizer)
    else:
        words, sub_words, keys = _tokenize(graph_text.text, tokenizer)
    graph_text.words = words
    shap_values_pred_1st = shap_values_pred[0]
    shap_values_pred_2nd = shap_values_pred[1]
    orig_probs = target_model(data.x,data.edge_index)[graph_text.neighbor]
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs, -1)
    graph_text.pre_label = orig_label
    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    llm_mask_model = llm_mask_model.to(device)
    word_predictions = llm_mask_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, top_k, -1)
    ids2tokens = []
    for i in range(word_predictions.size(0)):  # seq-len k
        ids2tokens.append(tokenizer.convert_ids_to_tokens(word_predictions[i]))
    word_predictions = word_predictions[1:-1, :]
    word_pred_scores_all = word_pred_scores_all[1:-1, :]
    ids2tokens = ids2tokens[1:-1]
    
    #index
    word_shap_values = []
    if args.using_shap:
        for start, end in keys:
            word_shap_1st = sum(shap_values_pred_1st[i] for i in range(start, end))
            word_shap_2nd = sum(shap_values_pred_2nd[i] for i in range(start, end))
            word_shap_values.append(float(sum(word_shap_1st - word_shap_2nd)))
        word_shap_values = np.array(word_shap_values)  #  NumPy 
        sorted_indices = np.argsort(-word_shap_values)
        logger.info("Using Shapley Value")
    else:
        important_scores = get_important_scores(attack_idx, data, graph_text, target_model, llm_emb_model, orig_probs,orig_label,
                                                tokenizer, max_length,device)
        important_scores = np.array(important_scores)  #  NumPy 
        sorted_indices = np.argsort(-important_scores)
        logger.info("Using Important Score")

    original_text_emb = copy.deepcopy(data.x[attack_idx])
    final_words = copy.deepcopy(words)
    vul_nodes = graph_text.neighbor
    vul_mask = torch.ones(len(graph_text.neighbor),dtype=torch.bool)
    score_final = Score(int(attack_idx), 0 , 0)
    score_final.score_important = graph_text.attack_number

    edge_changed = 0
    for top_index in sorted_indices:
        score_final.score_normal = 0
        if graph_text.change > int(0.3 * (len(words))):
            graph_text.attack_words = final_words
            graph_text.text = " ".join(final_words)
            inputs = tokenizer(graph_text.orig_text)
            orig_feature = sentence2emb(graph_text.orig_text, tokenizer, llm_emb_model,device = device)
            logger.info("Orig_text : {}\n Cur_text : {}".format(graph_text.orig_text, graph_text.text))
            final_feature = sentence2emb(graph_text.text, tokenizer, llm_emb_model,device = device)
            final_feature = torch.tensor(final_feature).to(device)
            return final_feature,graph_text.text
        tgt_word = words[top_index]
        if tgt_word in filter_words:
            continue
        if tgt_word.isdigit():
            continue
        if tgt_word in string.punctuation:
            continue
        if keys[top_index][0] > max_length - 2:
            continue
        if len(tgt_word) < 2 and tgt_word.isalpha():
            continue
        substitutes = word_predictions[keys[top_index][0]:keys[top_index][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys[top_index][0]:keys[top_index][1]]
        
        substitutes = get_substitues(substitutes, tokenizer, word_pred_scores, 2.0)
        target_model.eval()
        logits = torch.softmax(target_model(data.x,data.edge_index)[graph_text.neighbor], -1)
        gap = 0
        candidate = None
        for substitute_ in substitutes:
            substitute = substitute_
            if args.llm == 'roberta' or args.llm == 'deberta':
                substitute = substitute.replace('Ġ','')
            if substitute.lower() == tgt_word.lower():
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word
            if substitute in filter_words:
                continue
            if substitute in string.punctuation:
                continue
            if not is_word(substitute):
                continue
            
            temp_replace = final_words.copy()
            temp_replace[top_index] = substitute
            temp_text = " ".join(temp_replace)
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length)
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)
            feature_ = sentence2emb(temp_text, tokenizer, llm_emb_model,device = device)
            feature_ = torch.tensor(feature_).to(device)
            data.x[attack_idx] = feature_
            all_prob = target_model(data.x,data.edge_index)
            temp_prob = torch.softmax(all_prob[graph_text.neighbor], -1)
            temp_label = torch.argmax(temp_prob, -1)

            logits_diff = logits - temp_prob 
            text_similarity = F.cosine_similarity(feature_, original_text_emb)
            
            graph_text.mask = ((graph_text.label.squeeze() == temp_label) | (graph_text.wrong_mask))
            graph_text.attack_nodes = graph_text.neighbor[~graph_text.mask]
            score_logits = torch.gather(logits_diff, 1, orig_label.unsqueeze(1)).squeeze(1)
            score_logits = score_logits[vul_mask].sum()
            # score_sim = 1 - text_similarity
            score = score_logits 
            unequal_num = (graph_text.label != temp_label.unsqueeze(1)).sum()

            if unequal_num >= graph_text.attack_number:

                if (unequal_num > score_final.score_important):
                    score_final.score_important = unequal_num
                    score_final.score_normal = score
                    score_final.word = substitute
                    graph_text.attack_number = unequal_num
                    candidate = substitute
                else:
                    if (score > score_final.score_normal):
                        score_final.score_normal = score
                        score_final.word = substitute
                        candidate = substitute
                if (graph_text.change >= int (0.25 * (len(words)))) & (temp_label[0] == data.y[attack_idx].item()) & (edge_changed == 0):
                    vul_nodes = find_vulnerable_neighbor(attack_idx, graph_text.neighbor, data.node_degree, temp_prob, P, device,graph = graph, topk = 20)
                    shap = GNNShapExplainer(target_model, data, nhops=2, verbose=0, device=device,progress_hide=True)
                    
                    explanation,vul_edge_index = shap.explain(attack_idx, nsamples=10000,neighbor = vul_nodes,graph = graph,
                                            sampler_name='GNNShapSampler', batch_size=512,
                                            solver_name='WLSSolver')
                    if vul_edge_index == None:
                        edge_changed = 1
                        continue
                    res = explanation.result2dict()
                    
                    res['scores'] = torch.tensor(res['scores']).to(device)
                    edge_top_k = min((res['scores'] > 0).size(0),4)
                    tmp_indices = torch.argsort(-res['scores'])[:edge_top_k]
                    tmp_indices = torch.tensor(tmp_indices).to(device)
                    if int(tmp_indices.max()) > vul_edge_index.size(1):
                        edge_changed = 1
                        continue
                    vul_edge_index = vul_edge_index[:,tmp_indices]
                    # reverse_edge_index = torch.stack([vul_edge_index[1], vul_edge_index[0]], dim=0)
                    # vul_edge_index = torch.cat([vul_edge_index, reverse_edge_index], dim=1)
                    # vul_nodes = torch.cat((vul_edge_index[0], vul_edge_index[1])).unique()
                    ori_edge_index = data.edge_index
                    data.edge_index = remove_edges(data.edge_index , graph, vul_edge_index)
                    prob = target_model(data.x, data.edge_index)[attack_idx]
                    # if torch.argmax(temp_prob, -1) == data.y[attack_idx]:
                    #     data.edge_index = ori_edge_index
                    logits_2nd = all_prob[vul_nodes]
                    pred_2nd = torch.topk(logits_2nd, k=2, dim=1, largest=True)
                    pred_2nd = pred_2nd.indices[:,1]
                    edge_changed = 1
                    
                
        if candidate:
            if args.llm == 'roberta' or args.llm == 'deberta':
                candidate = candidate.replace('Ġ','')
            logger.info("replace: {} with {}".format(final_words[top_index],candidate))
            final_words[top_index] = candidate
            graph_text.change += 1
            

                
    graph_text.attack_words = final_words
    graph_text.text = " ".join(final_words)
    logger.info("Orig_text : {}\n Cur_text : {}".format(graph_text.orig_text, graph_text.text))
    final_feature = sentence2emb(graph_text.text, tokenizer, llm_emb_model,device = device)
    orig_feature = sentence2emb(graph_text.orig_text, tokenizer, llm_emb_model,device = device)
    final_feature = torch.tensor(final_feature).to(device)
    
    return final_feature, graph_text.text

def main():
    parser = argparse.ArgumentParser(description='attack')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--llm',type=str,default='bert')
    parser.add_argument('--dataset',type=str,default='cora',choices=['cora', 'citeseer','pubmed'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inductive', default=True,action="store_true")
    parser.add_argument('--re_split', default=False,action="store_true")
    parser.add_argument('--using_shap', default=True,action="store_true")
    parser.add_argument('--mode',type=str,default='full')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--best_weights',default=True, action="store_true")
    parser.add_argument('--embedding', type=str, default='bert', choices=['bert', 'bow', 'gtr'])
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--llm_path', type=str, default='bert-base-uncased')
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.8)

    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to use for GNNShap')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sampler', type=str, default='GNNShapSampler',
                        help='Sampler to use for sampling coalitions',
                        choices=['GNNShapSampler', 'SVXSampler', 'SHAPSampler',
                                'SHAPUniqueSampler'])
    parser.add_argument('--solver', type=str, default='WLSSolver',
                        help='Solver to use for solving SVX', choices=['WLSSolver', 'WLRSolver'])
    

    #1. prepare hyper parameters, logger and dataset  #####
    args = parser.parse_args()
    logger_name = f'{args.dataset}_{args.llm}_attack'
    data_path = config.get_dataset_path(args.dataset, args.llm)
    model_weight_path = config.get_model_path(args.llm, args.dataset, 2, 200, args.lr, args.train_ratio, args.val_ratio, args.test_ratio)
    attack_nodes_path = config.get_attack_nodes_path(args.llm, args.dataset, 0)
    logger = create_logger(args, logger_name)
    set_seed(args.seed)
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    llm_model_path = config.get_llm_path(args.llm)
    
    if args.llm == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(llm_model_path)
        llm_mask_model = RobertaForMaskedLM.from_pretrained(llm_model_path).to(device)
        llm_emb_model = AutoModel.from_pretrained(llm_model_path).to(device)  
    elif args.llm == 'deberta': 
        tokenizer = DebertaTokenizer.from_pretrained(llm_model_path)
        llm_mask_model = DebertaForMaskedLM.from_pretrained(llm_model_path).to(device)
        llm_emb_model = AutoModel.from_pretrained(llm_model_path).to(device)  
    elif args.llm == 'distilbert': 
        tokenizer = DistilBertTokenizer.from_pretrained(llm_model_path)
        llm_mask_model = DistilBertForMaskedLM.from_pretrained(llm_model_path).to(device)
        llm_emb_model = AutoModel.from_pretrained(llm_model_path).to(device)  
    else:
        tokenizer = BertTokenizer.from_pretrained(llm_model_path)
        llm_mask_model = BertForMaskedLM.from_pretrained(llm_model_path).to(device)
        llm_emb_model = AutoModel.from_pretrained(llm_model_path).to(device)  

    dataset = tag_dataset(args,logger,device,data_path)
    data = dataset.process_data(tokenizer,llm_emb_model, data_path)
    texts = data.raw_texts
    data = data.to(device)

    model = GCN(data.num_features, args.hidden_channels,data.num_classes, args.num_layers,args.dropout)
    model = model.to(device)
    
    #2. Load model
    logger.info(f"Loading model from {model_weight_path}")
    w = torch.load(model_weight_path, weights_only=False)
    model.load_state_dict(w)
    # train_model(model, data, args)
    
    
    #3. find target nodes
    graph = nx.DiGraph()
    all_nodes = list(range(data.y.size(0)))
    edge_index_np = data.edge_index.cpu().numpy()
    edges = edge_index_np.T.tolist()
    edges_with_weight = [(u, v, {'weight': 1.0}) for u, v in edges]
    graph.add_nodes_from(all_nodes)
    graph.add_edges_from(edges_with_weight)
    with torch.no_grad():
        model.eval()
        logits = torch.softmax(model(data.x,data.edge_index),-1)
        pred = torch.argmax(logits,-1)
        pred_second = torch.topk(logits, k=2, dim=1, largest=True)
        pred_second = pred_second.indices[:,1]
    correct_mask = (pred == data.y.reshape(-1))
    
    # Load or generate attack nodes
    if os.path.exists(attack_nodes_path):
        attack_nodes = torch.load(attack_nodes_path, weights_only=False)
        logger.info(f"Loaded attack nodes from {attack_nodes_path}")
    else:
        attack_nodes = find_attack_nodes_random(data, correct_mask, device=device)
        torch.save(attack_nodes, attack_nodes_path)
        logger.info(f"Generated and saved attack nodes to {attack_nodes_path}")
    
    data.attack_idx = attack_nodes
    
    
    # calculate the original metric    
    orgi_f1 = test(model, data, data.test_indices)
    orgi_f1_ = test(model, data, data.attack_idx,logits)
    logger.info("Orgi_f1: {}\n Orgi_f1 for attack node: {}".format(orgi_f1, orgi_f1_))
    
    #4. find the neighbors and calculate the assortativity
    neighbor_nodes, weights = find_neighbors_with_weights(graph, attack_nodes,device=device)
    assortativity_dict = compute_assortativity(graph, attack_nodes, neighbor_nodes, pred)
    logger.info("assortativity_dict: {}".format(assortativity_dict))
    # add_heterophilous_edges(graph, data, assortativity_dict, attack_nodes, neighbor_nodes, pred_second, theta=0.6, min_nodes=5)
    neighbor_nodes, weights = find_neighbors_with_weights(graph, attack_nodes,device=device)
    new_assortativity_dict = compute_assortativity(graph, attack_nodes, neighbor_nodes, pred)
    logger.info("new_assortativity_dict: {}".format(new_assortativity_dict))
    data.node_degree = degree(data.edge_index[0,:])
  
    #5. find important words
    attack_text = [texts[idx] for idx in attack_nodes]
    attack_label = data.y[attack_nodes]
    target_model = model
    top_k = args.top_k
    batch_size = 32
    max_length = 512 
    num_classes = data.num_classes


    for index, text in enumerate(attack_text): 
        P_matrix = pagerank(graph, k=3)
        node_index = int(data.attack_idx[index])
        # neighbor = neighbor_nodes[index]
        neighbor = node_neighbor(node_index, data.edge_index, device,num_nodes=data.x.size(0))
        all_flattened_labels = []
        for node_id in neighbor:
            for label_idx in range(num_classes):
                all_flattened_labels.append(f"Node_{node_id}_Label_{label_idx}")

        # Create id2label and label2id for ALL flattened outputs
        id2label_flat = {i: label_name for i, label_name in enumerate(all_flattened_labels)}
        label2id_flat = {label_name: i for i, label_name in enumerate(all_flattened_labels)}

        # Assign these to your model's config
        llm_emb_model.config.id2label = id2label_flat
        llm_emb_model.config.label2id = label2id_flat
        if args.using_shap:
            #shap_values
            classifier  = pipeline(model=llm_emb_model, 
                                   gnn = model, 
                                   graph_data = deepcopy(data), 
                                   pipeline_class = TAG_Pipeline, 
                                   task = "feature-extraction", 
                                   tokenizer = tokenizer,
                                   index = node_index,
                                   neighbor = neighbor,
                                   device = device) 
            explainer = shap.Explainer(classifier)
            shap_values = explainer([text])
            shap_values.base_values = shap_values.base_values.reshape(-1,num_classes)
            category_1 = np.argmax(shap_values.base_values,-1)
            category_2 = pred_second[neighbor].cpu().numpy()

            shap_values_1 = shap_values.values[0].reshape(-1, len(neighbor),num_classes)
            shap_values_2 = shap_values.values[0].reshape(-1, len(neighbor),num_classes)

            shap_values_pred_1st = np.zeros((shap_values_1.shape[0]-2, len(neighbor), 1))
            shap_values_pred_2nd = np.zeros((shap_values_2.shape[0]-2, len(neighbor), 1))
            for i in range(len(neighbor)):
                shap_values_pred_1st[:, i, 0] = shap_values_1[1:-1, i, category_1[i]]
                shap_values_pred_2nd[:, i, 0] = shap_values_2[1:-1, i, category_2[i]]
            shap_values_pred = (shap_values_pred_1st, shap_values_pred_2nd)

        neighbor = torch.IntTensor(list(neighbor_nodes[index]))
        weight  = torch.Tensor(list(weights[index]))
        label = attack_label[index]
        text = text.strip()
        graph_text = Graph_Text(text, neighbor.to(device), weight, data.y[neighbor])
        graph_text.orig_text = text
        temp_label =  torch.argmax(model(data.x, data.edge_index)[neighbor],-1)
        graph_text.attack_number = (graph_text.label != temp_label.unsqueeze(1)).sum()
        graph_text.wrong_mask = (graph_text.label.squeeze() != temp_label)
        if args.using_shap:
            feat,text_ = attack(args, 
                          logger, 
                          attack_nodes[index], 
                          data, 
                          graph_text, 
                          target_model, 
                          llm_mask_model, 
                          llm_emb_model, 
                          tokenizer, 
                          top_k, 
                          batch_size,
                          index, 
                          shap_values_pred, 
                          device, 
                          graph = graph,
                          P = P_matrix)
        else:
            feat,text_ = attack(args, 
                          logger, 
                          attack_nodes[index], 
                          data, 
                          graph_text, 
                          target_model,
                          llm_mask_model, 
                          llm_emb_model, 
                          tokenizer, 
                          top_k, 
                          batch_size,
                          index, 
                          None,
                          device, 
                          max_length)
        before = pred[neighbor].sum()/neighbor.size(0)
        data.x[data.attack_idx[index]] = feat
        data.raw_texts[data.attack_idx[index]] = text_
        attack_pred_ = torch.softmax(model(data.x,data.edge_index),-1)
        attack_pred = torch.argmax(attack_pred_,-1)
        before_f1 = accuracy_score(pred[neighbor].cpu(), data.y[neighbor].cpu())
        after_f1  = test(model, data, neighbor)
        test_acc = test(model, data, data.test_indices)
        attack_node_acc = test(model, data, data.attack_idx)
        logger.info("Node: {}".format(data.attack_idx[index]))
        logger.info("Success: {}".format((pred[data.attack_idx[index]]== data.y[data.attack_idx[index]]) and (not attack_pred[data.attack_idx[index]]== data.y[data.attack_idx[index]])))
        change_node_index = torch.where(~(pred[neighbor] == attack_pred[neighbor]))
        logger.info("This node before : {}, After: {}".format(before_f1,after_f1))
        logger.info("Test_acc: {}".format(test_acc))
        logger.info("Attack_node_acc: {}".format(attack_node_acc))
    
    cur_f1 = test(model, data, data.test_indices)
    cur_f1_ = test(model, data, data.attack_idx)
    logger.info("Cur_f1: {}\n Cur_f1 for attack node: {}".format(cur_f1, cur_f1_))
    logger.info(data.edge_index.size())
    tmp_data = torch.load(data_path, weights_only=False).to(device)
    similarities = F.cosine_similarity(tmp_data.x[attack_nodes], data.x[attack_nodes], dim=1)
    logger.info(similarities)
    similarities = similarities.mean()
    logger.info(similarities)


    
    
if __name__ == "__main__":
    main()
