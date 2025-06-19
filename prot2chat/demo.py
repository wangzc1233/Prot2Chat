import os
import torch
import argparse
from flask import Flask, request, jsonify, render_template
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from modelscope import AutoModelForCausalLM, AutoTokenizer
import preprocess
import traceback  
from torch import nn
import math
from peft import LoraConfig, PeftModel
"""
python /data/zcwang/Protein/project/demo.py --model_path /data/zcwang/Protein/llama_output/output_v3 --lora_path /data/zcwang/Protein/project/2502/v12-2/lora_weight/lora_weights_1_400000.pth --adapter_path /data/zcwang/Protein/project/2502/v12-2/adapt_weight/adapter_model_and_optimizer_1_400000.pth --port 8888 --gpu 0,1,2,3
"""
# Dynamic positional encoding
class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create a constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe

# Protein structure sequence adapter
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
        
        self.question_proj = nn.Linear(output_dim, output_dim)
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)

    def forward(self, x, h_state):
        # x shape [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # Pad the input to max_len
        if seq_len < self.max_len:
            pad_size = self.max_len - seq_len
            padding = torch.zeros(batch_size, pad_size, self.input_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)  # Shape: [batch_size, max_len, input_dim]
        elif seq_len > self.max_len:
            x = x[:, :self.max_len, :]  # Truncate if seq_len > max_len
        
        # Project the input
        x_proj = self.linear_proj(x)  # Shape: [batch_size, max_len, output_dim]

        # Norm1
        x_proj = self.layer_norm1(x_proj)
        
        # Positional encoding
        pe = self.pos_encoder()

        # Apply positional encoding
        x_pos_encoded = x_proj + pe[:, :x_proj.size(1), :]  # Shape: [batch_size, max_len, output_dim]
        
        # Prepare learnable queries
        queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_queries, output_dim]

        h_state = self.question_proj(h_state)

        queries = queries + h_state
        # Norm2
        queries = self.layer_norm2(queries)

        # Apply positional encoding to queries
        queries_pos_encoded = pe[:, :self.num_queries*2:2, :] + queries  # Shape: [batch_size, num_queries, output_dim]
        
        # Cross-attention
        attn_output, _ = self.cross_attention(queries_pos_encoded, x_pos_encoded, x_pos_encoded)  # Shape: [batch_size, num_queries, output_dim]
        
        # Apply layer normalization
        attn_output = self.layer_norm3(attn_output)
        
        # Project the output
        output = self.output_proj(attn_output)  # Shape: [batch_size, num_queries, output_dim]
        
        return output

# Global variables to store the loaded model and adapter
model = None
tokenizer = None
adapter = None
device = None

# Initialize the model and adapter
def initialize_models(model_path, lora_path, adapter_path):
    global model, tokenizer, adapter, device
    
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model L (LLaMA model)
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # Load LoRA weights
    print(f"Loading LoRA weights: {lora_path}")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    # Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load adapter model A
    print(f"Loading adapter model: {adapter_path}")
    adapter = ProteinStructureSequenceAdapter(input_dim=1152, output_dim=4096, num_heads=16, num_queries=256, max_len=512).to(device)
    checkpoint = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(checkpoint['adapter_model_weight'])
    adapter.eval()
    
    print("All models loaded successfully")

# Main function: Process PDB file and question, generate answer
def generate_answer(pdb_file_path, question):
    global model, tokenizer, adapter, device
    
    # Ensure the models are loaded
    if model is None or tokenizer is None or adapter is None:
        return "Error: Models are not initialized"
    
    try:
        # Step 1: Process the PDB file to get the protein multi-dimensional vector
        protein_vector = preprocess.get_mpnn_emb(pdb_file_path).unsqueeze(0).to(device)
        
        # Ensure the shape of the protein vector is correct
        max_length = 512
        if max_length > protein_vector.size(1):
            padding_length = max_length - protein_vector.size(1)
            padding = torch.zeros((1, padding_length, 1152), device=device)
            protein_vector = torch.cat([protein_vector, padding], dim=1)
        else:
            protein_vector = protein_vector[:, :max_length, :]
        
        # Step 2: Get the hidden state of the question text
        inputs = tokenizer(f"Human: {question}\nAssistant: ", return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True).hidden_states[-1]
        question_hidden_state = hidden_states[:, -1, :]
        
        # Step 3: Input the protein vector and question hidden state into the adapter model to get the new protein embedding
        with torch.no_grad():
            protein_embedding = adapter(protein_vector, question_hidden_state)
        
        # Step 4: Use the embedding layer of the model to encode the question text
        inputs_embeds = model.base_model.model.model.embed_tokens(inputs.input_ids)
        
        # Step 5: Concatenate the protein embedding and text embedding, input into the model to get the answer
        combined_embeds = torch.cat([protein_embedding, inputs_embeds], dim=1)
        combined_attention_mask = torch.cat([torch.ones((protein_embedding.size(0), protein_embedding.size(1)), device=device), inputs.attention_mask], dim=1)
        
        # Generate the answer
        with torch.no_grad():
            generated_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128
            )
        
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the Assistant part of the answer
        if "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()
        
        return response
    except Exception as e:
        # Capture and print detailed error information
        print(f"Error in generate_answer: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing: {str(e)}"

# Create a Flask application
def create_app():
    app = Flask(__name__)
    
    # Create a simple HTML template
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prot2Chat: Protein Q&A System</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input[type="file"], input[type="text"] { width: 100%; padding: 8px; }
                button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
                #loading { display: none; margin-top: 20px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>Prot2Chat: Protein Q&A System</h1>
            <form id="queryForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="pdbFile">Upload PDB file:</label>
                    <input type="file" id="pdbFile" name="pdbFile" accept=".pdb" required>
                </div>
                <div class="form-group">
                    <label for="question">Enter your question:</label>
                    <textarea id="question" name="question" placeholder="Please enter your question about protein..." required style="width: 100%; height: 100px;"></textarea>
                </div>
                <button type="submit">Submit</button>
            </form>
            <div id="loading">Processing, please wait...</div>
            <div id="result" style="display: none;">
                <h2>Answer:</h2>
                <p id="answer"></p>
            </div>

            <script>
                document.getElementById('queryForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    formData.append('pdbFile', document.getElementById('pdbFile').files[0]);
                    formData.append('question', document.getElementById('question').value);
                    
                    document.getElementById('result').style.display = 'none';
                    document.getElementById('loading').style.display = 'block';
                    
                    try {
                        const response = await fetch('/api/query', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        document.getElementById('loading').style.display = 'none';
                        
                        if (data.error) {
                            document.getElementById('answer').textContent = "Error: " + data.error;
                        } else {
                            document.getElementById('answer').textContent = data.answer;
                        }
                        document.getElementById('result').style.display = 'block';
                    } catch (error) {
                        document.getElementById('loading').style.display = 'none';
                        console.error('Error:', error);
                        document.getElementById('answer').textContent = "An error occurred, please try again: " + error;
                        document.getElementById('result').style.display = 'block';
                    }
                });
            </script>
        </body>
        </html>
        '''

    # API endpoint
    @app.route('/api/query', methods=['POST'])
    def query():
        try:
            # Get the uploaded PDB file
            if 'pdbFile' not in request.files:
                return jsonify({'error': 'No PDB file uploaded'}), 400
                
            pdb_file = request.files['pdbFile']
            if pdb_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
                
            question = request.form.get('question', '')
            if not question:
                return jsonify({'error': 'Question cannot be empty'}), 400
            
            # Save the PDB file to a temporary location
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdb_path = os.path.join(temp_dir, pdb_file.filename)
            pdb_file.save(temp_pdb_path)
            
            print(f"Processing file: {temp_pdb_path}")
            print(f"Question: {question}")
            
            # Generate the answer
            answer = generate_answer(temp_pdb_path, question)
            
            # Delete the temporary file
            if os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)
            
            return jsonify({'answer': answer})
        except Exception as e:
            print(f"API error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Protein Structure Q&A System')
    parser.add_argument('--model_path', type=str, default="/data/zcwang/Protein/llama_output/output_v3", 
                        help='Path to the LLaMA model')
    parser.add_argument('--lora_path', type=str, default="/data/zcwang/Protein/project/2502/v12-2/lora_weight/lora_weights_1_400000.pth", 
                        help='Path to the LoRA weights')
    parser.add_argument('--adapter_path', type=str, default="/data/zcwang/Protein/project/2502/v12-2/adapt_weight/adapter_model_and_optimizer_1_400000.pth", 
                        help='Path to the adapter model weights')
    parser.add_argument('--port', type=int, default=7777, 
                        help='Server port')
    parser.add_argument('--gpu', type=str, default="0,1", 
                        help='IDs of the GPUs to use, e.g., "0,1"')
    
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    
    initialize_models(args.model_path, args.lora_path, args.adapter_path)
    
    
    app = create_app()
    app.run(host='localhost', port=args.port, debug=False)