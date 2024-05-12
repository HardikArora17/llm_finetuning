from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def upload_to_hub(base_model_name, saved_model_name, output_model_name):
  model  = AutoModelForCausalLM.from_pretrained(saved_model_name, device_map='auto', trust_remote_code=True)  
  tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  
  model.push_to_hub(output_model_name)
  tokenizer.push_to_hub(output_model_name)

if __name__ == '__main__':
  base_model_name  =''
  saved_model_name = 'microsoft/phi-2'
  output_model_name = 'KnightHardik/astrophi-full'

  upload_to_hub(base_model_name, saved_model_name, output_model_name)
   