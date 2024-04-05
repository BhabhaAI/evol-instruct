import json
import random

from datasets import load_dataset
from models import Gemini, ChatGPT
import pandas as pd
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt


### INPUT PARAMETERS
dataset_name = 'yahma/alpaca-cleaned'  # Paste HF Dataset name
api_key = 'AIzaSyCFqknsY8TM3l4qfL0EsoPuQdldQ2HcjCE'  
model = Gemini(api_key)  # USE either 'Gemini' or 'ChatGPT'


dataset = load_dataset(dataset_name, split='train')
df = pd.DataFrame(dataset)

df['temp'] = df['instruction'].str.strip() + '\r\n'+ df['input'].str.strip()
evol_objs = []


for instruction in df['temp'][:10]:
	evol_prompts = []

	evol_prompts.append(createConstraintsPrompt(instruction))
	evol_prompts.append(createDeepenPrompt(instruction))
	evol_prompts.append(createConcretizingPrompt(instruction))
	evol_prompts.append(createReasoningPrompt(instruction))
	evol_prompts.append(createBreadthPrompt(instruction))

	selected_evol_prompt = random.choice(evol_prompts)

	evol_instruction = model.call_api(selected_evol_prompt)
	answer = model.call_api(evol_instruction)
	object = {"instruction":evol_instruction,"output":answer}
	evol_objs.append(object)

with open('new_dataset.json', 'w') as f:	
	json.dump(evol_objs, f, indent=4)




