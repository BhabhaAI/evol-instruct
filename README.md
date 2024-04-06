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

### API Endpoint

Query Parameters

1. `model` - Model name (either `gemini` or `chatgpt`)
2. `key` - Api Key
3. `dataset` - Dataset name in Huggingface Format
4. `instruction` - Name of the dataset's instruction prompt column
5. `input` (Optional) - Name of the dataset's input prompt column.
6. `rows` (Optional) - Number of rows to be generated.
7. `split` (Optional) - Dataset split. Defaults to `train`.

### Run API

1. Flask - `flask run`
2. FastAPI - `uvicorn fast:app --reload`
