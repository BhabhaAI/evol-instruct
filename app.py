import json
import random

from datasets import load_dataset
from flask import Flask, request, jsonify
from models import Gemini, ChatGPT
import pandas as pd
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt

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
    if not rows:
        rows = -1
    else:
        rows = int(rows)

    
    model = model_map[model_name](key)
    dataset = load_dataset(dataset_name, split='train')
    
    df = pd.DataFrame(dataset)
    
    if input:
        df['temp'] = df[instruction].str.strip() + '\r\n'+ df[input].str.strip()
    else:
        df['temp'] = df[instruction].str.strip()
    
    evol_objs = []

    for instruction in df['temp'][:rows]:
        evol_prompts = [createConstraintsPrompt(instruction),
                        createDeepenPrompt(instruction),
                        createConcretizingPrompt(instruction),
                        createReasoningPrompt(instruction),
                        createBreadthPrompt(instruction)]

        selected_evol_prompt = random.choice(evol_prompts)

        evol_instruction = model.call_api(selected_evol_prompt)
        answer = model.call_api(evol_instruction)

        evol_objs.append({"instruction":evol_instruction,"output":answer})
        
    return jsonify(evol_objs)
    
if __name__=="__main__":
    app.run()
    
    
    
    
    