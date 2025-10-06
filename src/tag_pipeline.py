import torch
from transformers import Pipeline
class TAG_Pipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}  # preprocess 
        forward_kwargs = {}
        postprocess_kwargs = {}
        #  _forward 
        if "graph_data" in kwargs:
            forward_kwargs["graph_data"] = kwargs["graph_data"]
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        if 'gnn' in kwargs:
            forward_kwargs["gnn"] = kwargs["gnn"]
        if 'index' in kwargs:
            forward_kwargs["index"] = kwargs["index"]
        if 'neighbor' in kwargs:
            forward_kwargs["neighbor"] = kwargs["neighbor"]
            postprocess_kwargs ["neighbor"] = kwargs["neighbor"]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        inputs_ids = self.tokenizer(
                            inputs,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt",
                            max_length=512
                        )
        return {
            "model_input": inputs_ids,
        }

    def _forward(self, model_inputs,graph_data, gnn, index, neighbor):
        inputs = model_inputs["model_input"]
        bert_embeds = self.model(**inputs).last_hidden_state
        bert_embeds = bert_embeds.mean(1)
        if not isinstance(bert_embeds, torch.Tensor):
            feat = torch.tensor(bert_embeds, dtype=torch.float32).to(self.device)
        else:
            feat = bert_embeds.to(self.device)
        graph_data = graph_data.to(self.device)
        graph_data.x[index] = feat.squeeze()
        with torch.no_grad():
            pred = gnn(graph_data.x, graph_data.edge_index)[neighbor]
        model_outputs = {"logits": pred}
        return model_outputs
            

    def postprocess(self, model_outputs, neighbor,top_k=None, **kwargs):
        logits = model_outputs["logits"]
        # neighbor = model_outputs["neighbor"]
        probs = logits.softmax(-1)
        
        # print(f"Debug postprocess - logits shape: {logits.shape}")
        # print(f"Debug postprocess - neighbor length: {len(neighbor)}")
        
        results = []
        for neighbor_idx in range(probs.shape[0]):
            for class_idx in range(probs.shape[1]):
                results.append({
                    "label": f"Node_{neighbor[neighbor_idx]}_Label_{class_idx}",
                    "score": float(probs[neighbor_idx, class_idx])
                })
        
        # print(f"Debug postprocess - results length: {len(results)}")
        return results