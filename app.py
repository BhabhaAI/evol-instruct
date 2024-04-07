import random
from dotenv import load_dotenv 
from typing import Optional

from datasets import load_dataset
from fastapi import FastAPI, Depends
import pandas as pd
from pydantic import BaseModel

from models import Gemini, ChatGPT
from prompts import create_prompts

load_dotenv()
model_map = {'chatgpt' : ChatGPT, 'gemini' : Gemini}

class Params(BaseModel):
    model: str
    dataset: str
    instruction: str
    key: Optional[str] = None
    input: Optional[str] = None
    rows: Optional[int] = -1
    split : Optional[str] = 'train'

app = FastAPI()

@app.get("/")
def evol(params = Depends(Params)):
    model_name = params.model
    key = params.key
    dataset_name = params.dataset
    instruction = params.instruction
    input = params.input
    rows = params.rows
    split = params.split

    model = model_map[model_name](key)
    df = pd.DataFrame(load_dataset(dataset_name, split=split))
    
    if input:
        df['temp'] = df[instruction].str.strip() + '\r\n'+ df[input].str.strip()
    else:
        df['temp'] = df[instruction].str.strip()
    
    evol_dataset = []

    for instruction in df['temp'][:rows]:
        evol_prompts = create_prompts(instruction)
        selected_evol_prompt = random.choice(evol_prompts)

        evol_instruction = model.call_api(selected_evol_prompt)
        answer = model.call_api(evol_instruction)

        evol_dataset.append({"instruction":evol_instruction,"output":answer})
    return evol_dataset

@app.get('/dummy')
def evol(params=Depends(Params)):
    return params