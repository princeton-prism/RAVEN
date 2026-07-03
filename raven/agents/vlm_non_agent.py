import json
import numpy as np
import sys, os
import re
import base64, io
from PIL import Image
from time import strftime, localtime
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(sys.path[0] + '/..')
from raven.utils.util import file_to_string
from raven.agents.agent import Agent, AgentOutput
from raven.memory.memory import Memory
from raven.memory.video_memory import VideoMemory, ImageMemoryItem
from raven.memory.memory import VLMMemoryItem
from raven.utils.util import json_fixer


def parse_json(string):
    string = re.sub(r'\{+', '{', string)
    string = re.sub(r'\}+', '}', string)
    
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    parsed = json.loads(parsed)
    return parsed

def np_image_to_base64(image):

    image = Image.fromarray(np.uint8(image))
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def img_to_base64(img):
    if isinstance(img, np.ndarray):
        return np_image_to_base64(img)
    elif isinstance(img, str):
        # for navqa pickle file
        if img.endswith('.pkl'):
            # Load pickle file and extract image data
            import pickle
            with open(img, "rb") as f:
                pkl_data = pickle.load(f)
            
            # Extract image from pickle data (assuming it's a dictionary with 'image' key)
            if isinstance(pkl_data, dict) and 'cam0' in pkl_data:
                image_data = pkl_data['cam0']
                if isinstance(image_data, np.ndarray):
                    return np_image_to_base64(image_data)
                else:
                    raise ValueError(f"Unexpected image data type in pickle: {type(image_data)}")
            else:
                raise ValueError(f"Pickle file does not contain expected image data: {list(pkl_data.keys()) if isinstance(pkl_data, dict) else type(pkl_data)}")
        else:
            # Handle regular image files
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    else:
        raise ValueError("Unsupported image type for base64 conversion.")

def construct_messages_from_memory(memory_items: ImageMemoryItem | VLMMemoryItem, start_time=0, max_images=3000):

    messages = []
    
    # Limit the number of images to avoid API limits
    if len(memory_items) > max_images:
        # Sample evenly across the memory items
        step = len(memory_items) // max_images
        sampled_items = memory_items[::step][:max_images]
    else:
        sampled_items = memory_items
    
    for i, item in enumerate(sampled_items):
            t = item.time + start_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            text = f"Frame {i} at time={t}, the robot was at an average position of {np.array(item.position).round(3).tolist()}"
            text += f" with an average orientation of {round(item.theta, 3)} radians."

            text += " Image robot saw:"
            if hasattr(item, 'image'):
                base64_image = img_to_base64(item.image)
            elif hasattr(item, 'image_file_path'):
                base64_image = img_to_base64(item.image_file_path)
            else:
                raise ValueError("ImageMemoryItem has no image or image_file_path attribute.")
            
            text_content = {
                "type": "text", 
                "text": text
            }

            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": 'low',
                },
            }
            messages.append(text_content)
            messages.append(image_content)
            print(f"[{i}/{len(sampled_items)} {i/len(sampled_items)*100:.2f}%] Added {len(text_content)} text messages and {len(image_content)} image messages")

    print(f"Added {len(messages)//2} text and image messages")

    return HumanMessage(content=messages)


DEFAULT_QUESTION_MESSAGE = """```json
        {{
            "type_reasoning": "-input your reasoning in here for the type of question-", 
            "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
            "answer_reasoning", "-input your reasoning in here for the answer. If you do not know the answer, provide your best guess for the answer type you provide.-", 
            "text": "--a text answer here--",
            "binary": "yes/no",
            "position": "[x,y,z]",
            "orientation": "[-.92]", 
            "time": "HH:MM:SS",
            "duration": "2.4",
        }}
        ```
        """

class VLMNonAgent(Agent):
    def __init__(self, 
                 llm_type='llama3', 
                 num_ctx=8192, 
                 num_gen_tokens=16384, 
                 temperature=0, 
                 question_message=None, 
                 prompt_dir="unknown", 
                 max_message_length=1500):
        self.question_message = question_message if question_message is not None else DEFAULT_QUESTION_MESSAGE
        self.llm_type = llm_type
        self.chat = self.llm_selector(llm_type, temperature, num_ctx, num_gen_tokens)
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.prompt = file_to_string(top_level_path+f'{prompt_dir}/non_agent_system_prompt.txt')
        self.max_message_length = max_message_length


    def llm_selector(self, llm_type, temperature, num_ctx, num_gen_tokens):
        llm = None
        # Support for LLM Gateway
        if 'gpt' in llm_type:
            llm = ChatOpenAI(model=llm_type, temperature=temperature, max_tokens=num_gen_tokens)

        elif 'gemini' in llm_type:
            llm = ChatGoogleGenerativeAI(model=llm_type, temperature=temperature, max_output_tokens=num_gen_tokens)
        
        elif ('qwen' in llm_type and 'vl' in llm_type) or ('gemma3' in llm_type.lower()):
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx, base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        
        else:
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm


    def set_memory(self, memory: VideoMemory):
        self.memory = memory


    def query(self, question: str) -> AgentOutput:

        system_message = self.prompt
        memory_human_messages = construct_messages_from_memory(self.memory.get_working_memory(), 
                                                               start_time=self.memory.start_time, 
                                                               max_images=self.max_message_length)

        question_message = "Be sure to respond in the following format: \n\n"

        question_message += self.question_message
        question_message += f"Given the context above, please answer the question: {question}\n\n Follow the correct output format."


        inputs = [
            SystemMessage(content=system_message),
            memory_human_messages, 
            HumanMessage(content=question_message)
        ]

        while True:

            response = self.chat.invoke(inputs)
            response = ''.join(response.content.splitlines())
            print(response)
            try:
                response = json_fixer(response)
                if '```json' not in response:
                    # try parsing on its own since we cannot always trust llms
                    parsed = json.loads(response) 
                else:
                    parsed = parse_json(response)

                # then check it has all the required keys
                keys_to_check_for = ["time", "text", "binary", "position"]
                for key in keys_to_check_for:
                    if key not in parsed:
                        print(parsed)
                        raise ValueError("Missing all the required keys during generate. Retrying...")
                response = AgentOutput.from_dict(parsed)

            except Exception as e:
                print(response)
                print(e)
                print("Generate call failed. Retrying...")
                continue

            break


        return response