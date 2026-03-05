from pydantic import BaseModel, ConfigDict, Field
from typing import Dict


class JsonCalling(BaseModel):
    """Schema for a user natural language prompt.

    Attributes:
        prompt: The command to be converted into a function call.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class PropertyInfo(BaseModel):
    """Metadata for a function parameter.

    Attributes:
        type: Data type of the property (e.g., 'string', 'int').
    """
    type: str


class ReturnInfo(BaseModel):
    """Metadata for a function's return value.

    Attributes:
        type: Data type returned by the function.
    """
    type: str


class JsonDefinition(BaseModel):
    """Schema for function tool definitions used by the LLM.
    Attributes:
        name: Function identifier.
        description: Explanation of what the function does.
        parameters: Mapping of argument names to their types.
        returns: Information about the function's output.
    """
    name: str
    description: str
    parameters: Dict[str, PropertyInfo]
    returns: ReturnInfo
