import random

from datasets import load_dataset
from models import Gemini, ChatGPT
import pandas as pd
from prompts import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt

### INPUT PARAMETERS
dataset_name = 'yahma/alpaca-cleaned'  # Paste HF Dataset name
api_key = ''  
# USE either 'Gemini' or 'ChatGPT' - Uncomment to use the model
model = Gemini(api_key)  
# model = ChatGPT(api_key)

dataset = load_dataset(dataset_name, split='train')
df = pd.DataFrame(dataset)

### CONFIGURE INSTRUCTION / INPUT PROMPTS COLUMN
df['temp'] = df['instruction'].str.strip() + '\r\n'+ df['input'].str.strip()

evol_objs = []

### Preprocessing 
df['evol1'] = df['temp'].apply(createConstraintsPrompt)
df['evol2'] = df['temp'].apply(createDeepenPrompt)
df['evol3'] = df['temp'].apply(createConcretizingPrompt)
df['evol4'] = df['temp'].apply(createReasoningPrompt)
df['evol5'] = df['temp'].apply(createBreadthPrompt)

def random_choice(row):
    return random.choice([row['evol1'], row['evol2'], row['evol3'], row['evol4'], row['evol5']])

df['evol'] = df.apply(random_choice, axis=1)

### Synthetic Generation

outdf = pd.DataFrame({'instruction':[], 'answer':[]})
outdf['instruction'] = df['evol'].apply(model.call_api)
outdf['answer'] = outdf['instruction'].apply(model.call_api)
outdf.to_json('new_dataset.json', indent = 4)


