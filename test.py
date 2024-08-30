import os, sys, re, json
import codecs
from collections import defaultdict
from openai import OpenAI

from tap import Tap
import base64


class Args(Tap):
    prompt: str = "Input questions."
    image_path: str = "test01.png"

class LLM_mode:
    def __init__(self, args: Args):
        self.args: Args = args
        
        self.prompt: str = args.prompt
        self.image_path: str = args.image_path
        self.context_messages = []

        self.api_key = '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'

        
    def encode_image(x, image_path: str):
        with open(image_path, "rb") as image_file:
            #print(x)
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_image

    def run_GPT(self):
        ## 初期設定
        client = OpenAI(api_key='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') # newest key
        model_name = "gpt-4o"   # "gpt-4-1106-preview' #'gpt-3.5-turbo-1106"　#gpt-3.5-turbo-0125

        if len(self.image_path) > 0:
            base64_image = self.encode_image(self.image_path)
            ## プロンプト
            messages = [{
                         "role": "system", 
                         "content":  "you are a sophisticated customer service staff."
                        },
                        {"role": "user", 
                         "content": [{"type": "text", "text": self.prompt},
                                     {"type": "image_url",
                                      "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                    ]
                        }]
        else:
            ## プロンプト
            messages = [{
                         "role": "system", 
                         "content":  "you are a sophisticated customer service staff."
                        },
                        {"role": "user", 
                         "content": [{"type": "text", "text": self.prompt}]
                        }]

        messages.extend(self.context_messages)
        response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0)
        response_message = response.choices[0].message.content
        #output = res['choices'][0]['message']['content']
        output = response_message

        self.context_messages.append({
                            'role': 'assistant',
                            'content': output,
        })

        print(output)
        return output

def main(args: Args):
    play = LLM_mode(args=args)
    play.run_GPT()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)