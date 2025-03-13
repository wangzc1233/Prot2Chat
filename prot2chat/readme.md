Welcome to Prot2Chat repository!

Paper: https://arxiv.org/abs/2502.06846

Prepare ProteinMPNN as shown in prot2chat/ProteinMPNN_m/proteinmpnn.md

bash prot2chat/requirement.sh

Download base_model and adapter_checkpoint from  

python generate.py  --base_model_path=your_path_to_base_model --lora_weight_path=prot2chat/lora_wight/lora_weight.pth --pdb_path=prot2chat/pdbs --checkpoint_path=your_path_to_adapter_checkpoint --conversation_data_path=prot2chat/QA_data/test.json
