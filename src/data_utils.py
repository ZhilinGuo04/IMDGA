import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import torch_geometric.transforms as T
import numpy as np
from transformers import AutoModel
from transformers import BertConfig, BertTokenizer,BertForMaskedLM

class tag_dataset:
    def __init__(self,args,logger,device, path):
        self.args = args
        self.logger = logger
        self.device = device
        self.data = torch.load(path, weights_only=False)
    def split_data(self,path = None):
        # Use existing split from dataset if available
        if hasattr(self.data, 'train_mask') and hasattr(self.data, 'val_mask') and hasattr(self.data, 'test_mask'):
            train_mask = self.data.train_mask
            val_mask = self.data.val_mask
            test_mask = self.data.test_mask
        else:
            # Generate new split if not available
            _,train_mask, val_mask, test_mask = generate_split(self.data,self.args.train_ratio,self.args.val_ratio,self.args.test_ratio)
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask
        
        # Ensure indices are set
        if not hasattr(self.data, 'train_indices') or not hasattr(self.data, 'val_indices') or not hasattr(self.data, 'test_indices'):
            self.data.train_indices = torch.nonzero(train_mask, as_tuple=True)[0]
            self.data.val_indices = torch.nonzero(val_mask, as_tuple=True)[0]
            self.data.test_indices = torch.nonzero(test_mask, as_tuple=True)[0]

    def process_data(self, tokenizer, llm, save_path):
        edge_index = self.data.edge_index
        self.data = T.ToSparseTensor()(self.data)
        self.data.edge_index = edge_index
        self.data.num_classes = self.data.y.max().item() + 1
        
        if os.path.exists(save_path):
            self.data = torch.load(save_path, weights_only=False)
            if self.args.re_split == 'True':
                self.split_data()
                torch.save(self.data, save_path)
            x = text2emb(self.data.raw_texts, tokenizer, llm, device = self.device)
        else:
            self.data.x = text2emb(self.data.raw_texts, tokenizer, llm, device = self.device)
            self.data.y = self.data.y.squeeze()
            if isinstance(self.data.y, np.ndarray):
                self.data.text_len = [len(self.data.raw_texts[i].split()) for i in range(self.data.y.shape[0])]
            else:
                self.data.text_len = [len(self.data.raw_texts[i].split()) for i in range(self.data.y.size(0))]
            self.data.text_len = torch.tensor(self.data.text_len)
            self.split_data()
            torch.save(self.data, save_path)
        return self.data
    
        
def generate_split(data,train_ratio=0.1, val_ratio=0.1, test_ratio=0.8):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_mask = indices[:int(train_ratio * num_nodes)]
    val_mask = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
    test_mask = indices[int((train_ratio + val_ratio) * num_nodes):]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_mask] = 1
    data.val_mask[val_mask] = 1
    data.test_mask[test_mask] = 1

    data.train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()

    return data,data.train_mask,data.val_mask,data.test_mask


def text2emb(texts,tokenizer, model,device, batch_size = 32):
    all_embeds = []
    model.eval()
    for i in range(0, len(texts),batch_size):
        text = texts[i:i+batch_size]
        inputs = tokenizer(
            text,
            padding = 'max_length',
            truncation=True,
            return_tensors="pt",
            max_length= 512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            bert_embeds = outputs.last_hidden_state 
            bert_embeds = bert_embeds.mean(1)
        all_embeds.append(bert_embeds)

    feat = torch.cat(all_embeds, dim=0)  
    return feat