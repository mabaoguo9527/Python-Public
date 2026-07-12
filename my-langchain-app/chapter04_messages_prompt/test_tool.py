from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import (
    _convert_from_v1_to_chat_completions,
    _convert_message_to_dict,
)
from dotenv import load_dotenv
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)


class DeepSeekChatWithReasoning(ChatDeepSeek):
    """让 thinking 模式下含有 tool_calls 的 assistant 消息能回传 reasoning_content。

    LangChain 的 _convert_message_to_dict 不会把任意 additional_kwargs 合并进请求体，
    因此重写 _get_request_payload，在生成请求体后把 additional_kwargs 里的
    reasoning_content 注入回对应的 assistant 消息。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        converted = []
        for m in messages:
            d = _convert_message_to_dict(
                _convert_from_v1_to_chat_completions(m)
                if isinstance(m, AIMessage)
                else m
            )
            rc = m.additional_kwargs.get("reasoning_content")
            if rc is not None:
                d["reasoning_content"] = rc
            converted.append(d)

        payload["messages"] = converted
        return payload


model = DeepSeekChatWithReasoning(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# 注意：reasoning_content 必须放在 additional_kwargs 里（顶层会被 LangChain 丢弃）。
ai_message = {
    "role": "assistant",
    "content": "",
    "additional_kwargs": {"reasoning_content": ""},
    "tool_calls": [{
        "name": "get_weather",
        "args": {"location": "北京"},
        "id": "call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s",
        "type": "tool_call",
    }]
}

tool_message = {
    "role": "tool",
    "content": "今天北京天气晴朗，万里无云~",
    "tool_call_id": "call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s",
}

messages = [
    {"role": "user", "content": "北京天气如何"},
    ai_message,
    tool_message
]

response = model.invoke(messages)

print(response)
