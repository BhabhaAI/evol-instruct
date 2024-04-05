### Overview of Evol-Instruct

[Evol-Instruct](https://arxiv.org/abs/2304.12244) is a novel method using LLMs instead of humans to automatically mass-produce open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs. You can easily embark on your own evolutionary journey with the [Evol Script](https://github.com/omkar-334/Evol-Instruct).

Available Models - Gemini, GPT-3.5-Turbo

### Setup

1. Clone the repository or download as zip and extract it.

`git clone https://github.com/omkar-334/Evol-Instruct.git`

2. Create a virtual environment and activate it

`python -m venv .venv`

`.venv\Scripts\activate`

3. Install required libraries

`pip install -r requirements.txt`

### Run Code - `main.py`

1. Change Input Parameters  
   `dataset_name` - Enter Dataset name in HuggingFace format  
   `api_key` - Enter respective model api_key  
   `model` - Choose model (either Gemini or ChatGPT)  
2. Configure Instruction prompts as per the dataset  
