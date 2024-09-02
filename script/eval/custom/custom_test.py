import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from modelscope.hub.snapshot_download import snapshot_download
from PIL import Image
from tqdm import tqdm
import os
import warnings

# Disable logging for a cleaner output
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# Set device
device = 'cuda'  # or 'cpu'
torch.set_default_device(device)

# Model and tokenizer setup
model_name = '/home/new/桌面/bunny/Bunny/script/train/merged_model_conta_10'  # Choose the appropriate model
#model_name = '/home/new/桌面/bunny/Bunny/script/train/merged_model'  # Choose the appropriate model
offset_bos = 1  # Adjust this based on the specific model being used
# snapshot_download(model_id='thomas/siglip-so400m-patch14-384')
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # float32 for CPU
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model.eval()

# Load test data
with open("/home/new/桌面/bunny/Bunny/vqa-en/bunny_test_new.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = []

# Base path for images
base_image_path = "/home/new/桌面/bunny/Bunny"

# Process each item in the test data
for item in tqdm(test_data, desc="Processing items", unit="item"):
    # Update image path
    relative_image_path = item['image']
    image_path = os.path.join(base_image_path, relative_image_path.lstrip('./'))
    
    # Load the image
    image = Image.open(image_path)
    
    # Extract conversation details
    conversation = item['conversations']
    questions = [turn['value'] for turn in conversation if turn['from'] == 'human']
    answers = [turn['value'] for turn in conversation if turn['from'] == 'gpt']
    
    for i, question in enumerate(questions):
        # Prepare the input text
        input_text = f"Try to answer the questions in one english word or phrase： Human: <image>\n{question} gpt:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in input_text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)
        
        # Process image
        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=10,
                max_length=10,
                use_cache=True,
                repetition_penalty=1.5 # Adjust this as needed
            )[0]
        
        # Decode the generated response
        predicted_answer = tokenizer.decode(outputs[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Append results
        results.append({
            'question': question,
            'actual_answer': answers[i] if i < len(answers) else None,
            'predicted_answer': predicted_answer
        })

# Write results to file
with open('epoch10/conta/predict_new_conta.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
