import time
import requests
import os

from ratelimit import limits, sleep_and_retry
import google.generativeai as genai
import openai

class Gemini():
    RATE_LIMIT = 45
    PERIOD = 60

    def __init__(self, api_key=None, model_name="gemini-pro"):
        if api_key is None:
            api_key = os.getenv('GEMINI_KEY')

        # Configure generative AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(max_output_tokens=3000, temperature=0.0)
        self.safety_settings = {'HARASSMENT':'block_none', 'HATE_SPEECH': 'block_none', 'DANGEROUS': 'block_none', 'SEXUAL': 'block_none'}
        self.equal_rows = True

    @sleep_and_retry
    @limits(calls=RATE_LIMIT, period=PERIOD)
    def call_api(self, prompt, retry_count=0, max_retries=3):
        if retry_count < max_retries:
            try:
                response = self.model.generate_content(prompt, generation_config=self.generation_config, safety_settings=self.safety_settings)
                return response.text
            except Exception as e:

                retry_count += 1

                if "429" in str(e): 
                    print("429 error: Rate limit exceeded. Sleeping for 80 seconds.")
                    time.sleep(80)
                    return self.call_api(prompt, retry_count, max_retries)  # Retry the same request after sleeping  

                if "500" in str(e):
                    print("500 error: Internal server error. Sleeping for 80 seconds.")
                    time.sleep(80)
                    return self.call_api(prompt, retry_count, max_retries)  # Retry the same request after sleeping
                
                print(f"API call failed: {e}, retrying... Attempt {retry_count} of {max_retries}")
                return self.call_api(prompt, retry_count, max_retries)        
        else:
            print("Maximum retries reached. Moving on to the next item.")
            return None

class ChatGPT():
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        if api_key is None:
            api_key = os.getenv('CHATGPT_KEY')
        self.client = openai.OpenAI(api_key = api_key)
        
    def get_completion(self, prompt):
        try: 
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=2048,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)

            res = response.model_dump(mode='python')['choices'][0]['message']['content']
            return res
        except requests.exceptions.Timeout:
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            return None
        except openai.BadRequestError as e:
            # Handle the invalid request error here
            print(f"The OpenAI API request was invalid: {e}")
            return None
        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                print("The OpenAI API request timed out. Please try again later.")
    #             time.sleep(3)
                return self.get_completion(prompt)            
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                return None
        except openai.RateLimitError as e:
            return self.get_completion(prompt)

    def call_api(self, ins):
        success = False
        retry_count = 5
        ans = ''
        while not success and retry_count >= 0:
            retry_count -= 1
            try:
                ans = self.get_completion(ins)
                success = True
            except:
                time.sleep(5)
                print('retry for sample:', ins)
        return ans
        