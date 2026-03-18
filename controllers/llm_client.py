import os
from typing import Optional
from openai import OpenAI

class QwenAPIClient:
    """千问API调用客户端：封装通用的API调用逻辑"""
    def __init__(self, api_key: str, model_name: str = "qwen-plus", 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def call_qwen(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.1, max_tokens: int = 1000) -> Optional[str]:
        """
        调用千问API并返回结果
        :param system_prompt: 系统提示词
        :param user_prompt: 用户提示词
        :param temperature: 随机性（0~1）
        :param max_tokens: 最大返回长度
        :return: LLM返回文本（失败返回None）
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"千问API调用失败：{e}")
            print("参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
            return None