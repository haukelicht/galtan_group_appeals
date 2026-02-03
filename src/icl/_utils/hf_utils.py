from pydantic import BaseModel
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionInputResponseFormatJSONSchema, 
    ChatCompletionInputJSONSchema
)
from typing import Optional

def get_response_format(output_cls: BaseModel, output_cls_description: Optional[str] = None) -> ChatCompletionInputResponseFormatJSONSchema:
    json_schema = output_cls.model_json_schema()
    desc = output_cls_description or output_cls.__name__
    json_schema = ChatCompletionInputJSONSchema(name=desc, schema=json_schema)
    response_format = ChatCompletionInputResponseFormatJSONSchema(type="json_schema", json_schema=json_schema)
    return response_format
