import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  

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


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



device = "cuda" if torch.cuda.is_available() else "cpu"



model = AutoModelForCausalLM.from_pretrained(
    "/data/zcwang/Protein/llama_output/output_v3",
    torch_dtype=torch.float32,
    device_map="auto"
)


lora_wt='path/to/lora_weight'

#LoRA
config = LoraConfig(
    r=8, #16
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = PeftModel.from_pretrained(model, model_id=lora_wt,config=config)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True


tokenizer = AutoTokenizer.from_pretrained("/path/to/tokenizer")

lora_path='/path/to/save_lora_weight'

pdb_path = "/path/to/pdbfiles"
save_path = "path/to/save_adapt_weight"



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

        for conv in conversations:
            encoded_inputs = self.tokenizer("Human: %s\nAssistant: " % conv[0], return_tensors='pt',  max_length=self.max_length, truncation=True)
            input_ids = encoded_inputs.input_ids[0]
            attention_mask = encoded_inputs.attention_mask[0]
            x = len(encoded_inputs.input_ids[0])

            encoded_inputs = self.tokenizer("%s<|end_of_text|>\n" % (conv[1]), return_tensors='pt',  max_length=self.max_length, truncation=True)
            input_ids = torch.cat([input_ids, encoded_inputs.input_ids[0]], dim=0)
            attention_mask = torch.cat([attention_mask, encoded_inputs.attention_mask[0]], dim=0)

            target_mask = torch.zeros_like(input_ids)
            target_mask[x:-1] = 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            target_mask_list.append(target_mask)
        
        # Concatenate lists into tensors 
        input_ids_tensor = torch.cat(input_ids_list, dim=0)
        attention_mask_tensor = torch.cat(attention_mask_list, dim=0)
        target_mask_tensor = torch.cat(target_mask_list, dim=0)

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
            'target_mask': target_mask_tensor
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
        
        # Project the input  投影输入 
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

checkpoint = torch.load('path/to/checkpoint',map_location='cuda:0')

adapter = ProteinStructureSequenceAdapter(input_dim=1152, output_dim=4096, num_heads=16, num_queries=256, max_len=512).to(device)
adapter.load_state_dict(checkpoint['adapter_model_weight'])


for param in adapter.parameters():
    param.requires_grad = True

with open("/data/to/conversations.json", "r") as f:
    datasets = json.load(f)

random.shuffle(datasets)

# 
batch_size = 2
dataset = MultiModalDataset(datasets,  tokenizer=tokenizer, max_length=1024)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# 
criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.Adam([
	{'params': adapter.parameters(), 'lr': 0.0001,}, 
	{'params': model.parameters(), 'lr': 0.0001,},
	])
optimizer.load_state_dict(checkpoint['optimizer_weight'])

scaler = GradScaler()

max_length = 512 
num_epochs = 2
accumulation_steps = 16 


# 
for epoch in range(num_epochs):
    model.train()
    adapter.train()
    running_loss = 0.0
    start_time = time.time()  
    for batch_idx, batch in enumerate(dataloader):
        pdbs = batch["pdb"]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_mask = batch['target_mask'].to(device)
        
       
        batch_MPNN_outputs = []
        for pdb in pdbs:
            pdb_output = preprocess.get_mpnn_emb(os.path.join(pdb_path, pdb)).unsqueeze(0)  # (1, n, 1152)
            pdb_output = pdb_output.to(device)

            
            if max_length > pdb_output.size(1):
                padding_length = max_length - pdb_output.size(1)
                padding = torch.zeros((1, padding_length, 1152), device=device)
                pdb_output = torch.cat([pdb_output, padding], dim=1)
            else:
                pdb_output = pdb_output[:, :max_length, :]

            batch_MPNN_outputs.append(pdb_output)


     
        batch_MPNN_outputs = torch.cat(batch_MPNN_outputs, dim=0)  # (batch_size, 512, 1152)

        adapter_outputs = adapter(batch_MPNN_outputs) # (batch_size, 256, 4096)
       

        inputs_embeds = model.base_model.model.model.embed_tokens(input_ids)#这里改了

        
        inputs_embeds = torch.cat([adapter_outputs, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones((adapter_outputs.size(0), adapter_outputs.size(1)), device=device), attention_mask], dim=1)


        # print(numpy.shape(labels))
        labels = torch.where(target_mask == 1, input_ids, -100)
        # print(numpy.shape(labels))
        with autocast(device_type='cuda'):
            
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            
            
            logits = outputs.logits[:, adapter_outputs.size(1):]  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()


            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        
        scaler.scale(loss).backward()  
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  

        running_loss += loss.item()
        
        
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (batch_idx + 1)
        batches_left = len(dataloader) - (batch_idx + 1)
        epochs_left = num_epochs - (epoch + 1)
        time_left = avg_time_per_batch * batches_left + avg_time_per_batch * len(dataloader) * epochs_left

        # 将剩余时间转换为时:分:秒格式
        hours_left = int(time_left // 3600)
        minutes_left = int((time_left % 3600) // 60)
        seconds_left = int(time_left % 60)

        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
            f'Loss: {loss.item():.4f}, '
            f'Time per batch: {avg_time_per_batch:.2f} sec, '
            f'ETA: {hours_left:02d}:{minutes_left:02d}:{seconds_left:02d}')

        
        if (batch_idx + 1) % 5000 == 0:
            torch.save({
                'adapter_model_weight':adapter.state_dict(),
                'optimizer_weight':optimizer.state_dict()
            },save_path + '/' + f'adapter_model_and_optimizer_{epoch + 1}_{batch_idx + 1}.pth')
            print(f"Training complete and adapter model saved for epoch {epoch + 1} and step {batch_idx + 1}.")
            recent_lora_path = lora_path + '/' + f'lora_weights_{epoch + 1}_{batch_idx + 1}.pth'
            model.save_pretrained(recent_lora_path)
            tokenizer.save_pretrained(recent_lora_path)
            
            print(f"Training complete and llama model saved for epoch {epoch + 1} and step {batch_idx + 1}.")


        
        gc.collect()
        torch.cuda.empty_cache()

        
    
    
    average_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {average_loss:.4f}")

    
    torch.save(adapter.state_dict(), save_path + '/' + f'adapter_model_{epoch + 1}.pth')
    
    recentepoch_lora_path = lora_path + '/' + f'lora_weights_{epoch + 1}.pth'
    model.save_pretrained(recentepoch_lora_path)
    tokenizer.save_pretrained(recentepoch_lora_path)



# nohup  python adapter_train_v8.py  >my_train_log_1.out 2>&1 &

