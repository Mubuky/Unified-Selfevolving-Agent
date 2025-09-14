from typing import Callable, List
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage
import openai


class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool, base_url: str = None):
        self.model_name = llm_name
        if long_ver:
            llm_name = 'gpt-3.5-turbo-16k'
        
        llm_kwargs = {
            'model': llm_name,
            'temperature': 0.0,
            'openai_api_key': openai_api_key,
        }
        
        if base_url:
            llm_kwargs['base_url'] = base_url
            
        self.llm = ChatOpenAI(**llm_kwargs)

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        kwargs = {}
        if stop != []:
            kwargs['stop'] = stop
        for i in range(6):
            try:
                output = self.llm.invoke(messages, **kwargs).content.strip('\n').strip()
                break
            except openai.RateLimitError:
                print(f'\nRetrying {i}...')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output

def LLM_CLS(llm_name: str, openai_api_key: str, long_ver: bool, base_url: str = None) -> Callable:
    # 支持的模型列表（包括OpenAI兼容的模型）
    supported_models = [
        'gpt', 'deepseek', 'claude', 'llama', 'qwen', 'yi', 'glm', 
        'mixtral', 'gemini', 'claude-3', 'gpt-4', 'gpt-3.5'
    ]
    
    # 检查模型名是否包含支持的模型关键词
    if any(model_key in llm_name.lower() for model_key in supported_models):
        return GPTWrapper(llm_name, openai_api_key, long_ver, base_url)
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}. Supported models: {supported_models}")
