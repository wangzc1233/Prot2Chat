# Prot2Chat
Welcome to Prot2Chat repository!

Paper: https://arxiv.org/abs/2502.06846

STEP1、
Prepare ProteinMPNN as shown in prot2chat/ProteinMPNN_m/proteinmpnn.md

STEP2、
```shell
bash prot2chat/requirement.sh
```
STEP3、
Download base_model and adapter_checkpoint from  https://drive.google.com/file/d/1UgzkSda2wJew95FrVfF246IilB5ekTSc/view?usp=drive_link
STEP4、
```python
python generate.py  --base_model_path=your_path_to_base_model --lora_weight_path=prot2chat/lora_wight/lora_weight.pth --pdb_path=prot2chat/pdbs --checkpoint_path=your_path_to_adapter_checkpoint --conversation_data_path=prot2chat/QA_data/test.json
```
