import json
import random

from datasets import load_dataset
from models import Gemini, ChatGPT
import pandas as pd
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt

### INPUT PARAMETERS
dataset_name = 'yahma/alpaca-cleaned'  # Paste HF Dataset name
api_key = ''  
# USE either 'Gemini' or 'ChatGPT' - Uncomment to use the model
model = Gemini(api_key)  
# model = ChatGPT(api_key)


dataset = load_dataset(dataset_name, split='train')
df = pd.DataFrame(dataset)

### CONFIGURE INSTRUCTION PROMPTS
df['temp'] = df['instruction'].str.strip() + '\r\n'+ df['input'].str.strip()

evol_objs = []

for instruction in df['temp']:
	evol_prompts = [createConstraintsPrompt(instruction),
					createDeepenPrompt(instruction),
					createConcretizingPrompt(instruction),
					createReasoningPrompt(instruction),
					createBreadthPrompt(instruction)]

	selected_evol_prompt = random.choice(evol_prompts)

	evol_instruction = model.call_api(selected_evol_prompt)
	answer = model.call_api(evol_instruction)

	evol_objs.append({"instruction":evol_instruction,"output":answer})

with open('new_dataset.json', 'w') as f:	
	json.dump(evol_objs, f, indent=4)