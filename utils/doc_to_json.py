import inspect
from typing import Dict, Any, Callable
from ..tools.tool_annotation import *

def function_to_json(func: Callable) -> ToolType:
    """Converts a Python function to the specified JSON schema format."""
    PYTHON_TO_JSON_TYPES = {
        int: 'number',
        float: 'number',
        str: 'string',
        bool: 'boolean',
        type(None): 'null',
    }

    func_name = func.__name__
    docstring = inspect.getdoc(func) or ""
    description = docstring.strip() if docstring else ""

    sig = inspect.signature(func)
    parameters = sig.parameters

    properties: dict[str, ParameterProperty] = {}
    required: list[str] = []

    for param_name, param in parameters.items():
        if param_name == 'self':
            continue  # Skip 'self' in methods

        # Determine parameter type
        param_type = param.annotation
        json_type = 'string'  # Default if type is missing or unrecognized
        type_name = 'any'

        if param_type is not inspect.Parameter.empty:
            json_type = PYTHON_TO_JSON_TYPES.get(param_type, 'object')
            type_name = getattr(param_type, '__name__', str(param_type))
        else:
            type_name = 'any'

        # Generate parameter description
        param_description = f"{param_name}: {type_name}"

        # Check if required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

        # Build properties
        properties[param_name] = ParameterProperty(**{
            "type": json_type,
            "description": param_description
        })

    parameters_schema = Parameters(**{
        "type": "object",
        "properties": properties,
        "required": required if required else []
    })

    function_schema = FunctionDescription(**{
        "name": func_name,
        "description": description,
        "parameters": parameters_schema
    })

    return ToolType(**{
        "type": "function",
        "function": function_schema
    })

if __name__ == '__main__':
    import json

    # Example usage:
    def multiply(a: float, b: float) -> float:
        """Given two numbers a and b. Returns the product of them."""
        return a * b
    
    print(json.dumps(json.loads(function_to_json(multiply).model_dump_json()), indent = 2))