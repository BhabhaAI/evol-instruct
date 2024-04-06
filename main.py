import json
import random

from datasets import load_dataset
import pandas as pd

from models import Gemini, ChatGPT
from prompts import create_prompts

### INPUT PARAMETERS
dataset_name = 'yahma/alpaca-cleaned'  # Paste HF Dataset name
api_key = ''  
# USE either 'Gemini' or 'ChatGPT' - Uncomment to use the model
model = Gemini(api_key)  
# model = ChatGPT(api_key)

df = pd.DataFrame(load_dataset(dataset_name, split='train'))

### CONFIGURE INSTRUCTION PROMPTS
df['temp'] = df['instruction'].str.strip() + '\r\n'+ df['input'].str.strip()

evol_objs = []

for instruction in df['temp']:
	evol_prompts = create_prompts(instruction)

	selected_evol_prompt = random.choice(evol_prompts)

	evol_instruction = model.call_api(selected_evol_prompt)
	answer = model.call_api(evol_instruction)

	evol_objs.append({"instruction":evol_instruction,"output":answer})

with open('new_dataset.json', 'w') as f:	
	json.dump(evol_objs, f, indent=4)