import random

from datasets import load_dataset
from flask import Flask, request, jsonify
import pandas as pd

from models import Gemini, ChatGPT
from prompts import create_prompts

model_map = {'chatgpt' : ChatGPT, 'gemini' : Gemini}

app=Flask(__name__)

@app.route('/',methods=['GET'])
def evol_function():
    model_name = request.args.get('model')
    key = request.args.get('key')
    dataset_name = request.args.get('dataset')
    
    instruction = request.args.get('instruction')
    input = request.args.get('input')
    rows = request.args.get('rows')
    split = request.args.get('split')
    if not rows:
        rows = -1
    if not split:
        split = 'train'

    model = model_map[model_name](key)
    df = pd.DataFrame(load_dataset(dataset_name, split=split))
    
    if input:
        df['temp'] = df[instruction].str.strip() + '\r\n'+ df[input].str.strip()
    else:
        df['temp'] = df[instruction].str.strip()
    
    evol_dataset = []

    for instruction in df['temp'][:int(rows)]:
        evol_prompts = create_prompts(instruction)
        selected_evol_prompt = random.choice(evol_prompts)

        evol_instruction = model.call_api(selected_evol_prompt)
        answer = model.call_api(evol_instruction)

        evol_dataset.append({"instruction":evol_instruction,"output":answer})

    return jsonify(evol_dataset)
    
if __name__=="__main__":
    app.run(port = 8000)
    
    
    
    
    