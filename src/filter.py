from pydantic import BaseModel, ConfigDict, Field
from typing import Dict


class JsonCalling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class PropertyInfo(BaseModel):
    type: str


class ReturnInfo(BaseModel):
    type: str


class JsonDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, PropertyInfo]
    returns: ReturnInfo
