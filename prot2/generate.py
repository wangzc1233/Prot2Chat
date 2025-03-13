import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import argparse
import torch
import torch.nn as nn
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import preprocess
import json
import gc
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy
import math
import random
import time
from peft import LoraConfig, get_peft_model,PeftModel,TaskType,PeftConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction,corpus_bleu

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, required=True,
                    help="path_to_base_model")
parser.add_argument("--lora_weight_path", type=str, required=True,
                    help="path_to_lora_weights")
parser.add_argument("--pdb_path", type=str, required=True,
                    help='path_to_pdb_files')
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path_to_adapter_checkpoint")
parser.add_argument("--conversation_data_path", type=str, required=True,
                    help="path_to_QA_data")

args = parser.parse_args()


base_model_path =args.base_model_path # "path_to_base_model"
lora_weight_path =args.lora_weight_path # "path_to_lora_weights"
pdb_path =args.pdb_path # "path_to_pdb_files"
checkpoint_path =args.checkpoint_path # "path_to_adapter_checkpoint"
conversation_data_path =args.conversation_data_path # "path_to_QA_data"


model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)

lora_wt=lora_weight_path

#LoRA
config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = PeftModel.from_pretrained(model, model_id=lora_wt,config=config)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)


# 
class MultiModalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pdb = item['pdb']
        conversations = item['conversations']
        input_ids_list=[]
        attention_mask_list=[]
        target_mask_list=[]
        p_input_ids_list=[]
        p_attention_mask_list=[]
        for conv in conversations:
            encoded_inputs = self.tokenizer("Human: %s\nAssistant: " % conv[0], return_tensors='pt',  max_length=self.max_length, truncation=True)
            input_ids = encoded_inputs.input_ids[0]
            attention_mask = encoded_inputs.attention_mask[0]
            x = len(encoded_inputs.input_ids[0])
            a=input_ids
            b=attention_mask
            encoded_inputs = self.tokenizer("%s<|end_of_text|>\n" % (conv[1]), return_tensors='pt',  max_length=self.max_length, truncation=True)
            input_ids = torch.cat([input_ids, encoded_inputs.input_ids[0]], dim=0)
            attention_mask = torch.cat([attention_mask, encoded_inputs.attention_mask[0]], dim=0)

            target_mask = torch.zeros_like(input_ids)
            target_mask[x:-1] = 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            target_mask_list.append(target_mask)
            p_input_ids_list.append(a)
            p_attention_mask_list.append(b)
        # Concatenate lists into tensors 
        input_ids_tensor = torch.cat(input_ids_list, dim=0)
        attention_mask_tensor = torch.cat(attention_mask_list, dim=0)
        target_mask_tensor = torch.cat(target_mask_list, dim=0)
        p_input_ids_tensor = torch.cat(p_input_ids_list, dim=0)
        p_attention_mask_tensor = torch.cat(p_attention_mask_list, dim=0)

        # Padding to max_length
        current_len = input_ids_tensor.size(0)
        if current_len < self.max_length:
            pad_len = self.max_length - current_len
            input_ids_tensor = torch.cat([input_ids_tensor, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask_tensor = torch.cat([attention_mask_tensor, torch.zeros(pad_len, dtype=torch.long)])
            target_mask_tensor = torch.cat([target_mask_tensor, torch.zeros(pad_len, dtype=torch.long)])

        elif current_len > self.max_length:
            input_ids_tensor = input_ids_tensor[:self.max_length]
            attention_mask_tensor = attention_mask_tensor[:self.max_length]
            target_mask_tensor = target_mask_tensor[:self.max_length]

        return {
            'pdb': pdb,
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'target_mask': target_mask_tensor,
            'p_input_ids':p_input_ids_tensor,
            'p_attention_mask':p_attention_mask_tensor
       
        }
#
class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create constant positional encoding matrix with sin and cos 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe
#
class ProteinStructureSequenceAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_queries, max_len=512):
        super(ProteinStructureSequenceAdapter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.max_len = max_len
        
        # Linear projection layer 
        self.linear_proj = nn.Linear(input_dim, output_dim)
        
        # Dynamic positional encoding layer 
        self.pos_encoder = DynamicPositionalEncoding(output_dim, max_len)
        
        # Learnable queries 
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, output_dim))
        
        # Cross-attention layer 
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
        
        # Output projection layer 
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Layer normalization 
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x is of shape [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # Pad the input to max_len
        if seq_len < self.max_len:
            pad_size = self.max_len - seq_len
            padding = torch.zeros(batch_size, pad_size, self.input_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)  # Shape: [batch_size, max_len, input_dim]
        elif seq_len > self.max_len:
            x = x[:, :self.max_len, :]  # Truncate to max_len if seq_len > max_len
        
        # Project the input  
        x_proj = self.linear_proj(x)  # Shape: [batch_size, max_len, output_dim]

        # Norm1
        x_proj = self.layer_norm1(x_proj)
        
        # pe
        pe = self.pos_encoder()
        # print(numpy.shape(pe))

        # Apply positional encoding
        x_pos_encoded = x_proj + pe[:, :x_proj.size(1), :]  # Shape: [batch_size, max_len, output_dim]
        
        # Prepare learnable queries
        queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_queries, output_dim]

        # Norm2
        queries = self.layer_norm2(queries)
        # print(numpy.shape(queries))

        # Apply positional encoding to queries
        queries_pos_encoded = pe[:, :self.num_queries*2:2, :] + queries  # Shape: [batch_size, num_queries, output_dim]
        
        # Cross-attention
        attn_output, _ = self.cross_attention(queries_pos_encoded, x_pos_encoded, x_pos_encoded)  # Shape: [batch_size, num_queries, output_dim]
        
        # Apply layer normalization
        attn_output = self.layer_norm3(attn_output)
        
        # Project the output
        output = self.output_proj(attn_output)  # Shape: [batch_size, num_queries, output_dim]
        
        return output


checkpoint = torch.load(checkpoint_path,map_location='cuda:0')

adapter = ProteinStructureSequenceAdapter(input_dim=1152, output_dim=4096, num_heads=16, num_queries=256, max_len=512).to(device)
adapter.load_state_dict(checkpoint['adapter_model_weight'])


with open(conversation_data_path, "r") as f:
    datasets = json.load(f)


batch_size = 1
dataset = MultiModalDataset(datasets,  tokenizer=tokenizer, max_length=1024)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


max_length = 512 # 
num_epochs = 1
accumulation_steps = 16 


# 
for epoch in range(num_epochs):
    model.eval()
    adapter.eval()
    running_loss = 0.0
    scores=0.0
    count=0
    for batch_idx, batch in enumerate(dataloader):
        pdbs = batch["pdb"]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_mask = batch['target_mask'].to(device)
        p_input_ids=batch['p_input_ids'].to(device)
        p_attention_mask = batch['p_attention_mask'].to(device)
        
        batch_MPNN_outputs = []
        for pdb in pdbs:
            pdb_output = preprocess.get_mpnn_emb(os.path.join(pdb_path, pdb)).unsqueeze(0)  # (1, n, 1152)
            pdb_output = pdb_output.to(device)

            # Padding或Truncation
            if max_length > pdb_output.size(1):
                padding_length = max_length - pdb_output.size(1)
                padding = torch.zeros((1, padding_length, 1152), device=device)
                pdb_output = torch.cat([pdb_output, padding], dim=1)
            else:
                pdb_output = pdb_output[:, :max_length, :]

            batch_MPNN_outputs.append(pdb_output)

       
        batch_MPNN_outputs = torch.cat(batch_MPNN_outputs, dim=0)  # (batch_size, 512, 1152)

        adapter_outputs = adapter(batch_MPNN_outputs) # (batch_size, 256, 4096)

        inputs_embeds = model.base_model.model.model.embed_tokens(p_input_ids)

        inputs_embeds = torch.cat([adapter_outputs, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones((adapter_outputs.size(0), adapter_outputs.size(1)), device=device), p_attention_mask], dim=1)
        
        input_text = tokenizer.decode(p_input_ids[0], skip_special_tokens=True)
        print("Input text:",input_text)
       
        count+=1
        
        labels = torch.where(target_mask == 1, input_ids, 128000) 
        
        with torch.no_grad():
            
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            print(1)
            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                # do_sample=False,
                # num_beams=10
            )
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            # ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print('Generated text:',response)
           
            labels_decoded = tokenizer.decode(labels[0], skip_special_tokens=True)  
            print(f"Target text: {labels_decoded}")
            
            bleu_score=corpus_bleu([[labels_decoded.split()]],[response.split()],weights=[0.5, 0.5, 0, 0],smoothing_function=SmoothingFunction().method4)
            
        scores=scores+bleu_score
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{count}/{len(dataloader)}], '
            f'bleu_score: {bleu_score:.4f}'
            )
        
        gc.collect()
        torch.cuda.empty_cache()
    
    averange_score=scores/count
    print(f'Epoch [{epoch+1}/{num_epochs}],averange_score:{averange_score:.4f}')



# python generate.py  --base_model_path=your_path_to_base_model --lora_weight_path=your_path_to_lora_weights --pdb_path=/data/zcwang/prot2chat/pdbs --checkpoint_path=your_path_to_adapter_checkpoint --conversation_data_path=prot2chat/QA_data/test.json

