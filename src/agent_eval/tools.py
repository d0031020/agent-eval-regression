"""Two local tools. Arithmetic uses a bounded AST interpreter, never eval()."""

import ast
import json
import math
import operator
from importlib.resources import files


def lookup_order(order_id: str) -> dict:
    if not isinstance(order_id, str):
        raise ValueError("order_id must be a string")
    orders = json.loads(files("agent_eval").joinpath("fixtures/orders.json").read_text())
    if order_id not in orders:
        raise ValueError(f"unknown order: {order_id}")
    return {"order_id": order_id, **orders[order_id]}


def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
        raise ValueError("expression must be a string of 1 to 200 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression is too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("only numbers, parentheses, +, -, *, and / are supported")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("arithmetic magnitude must not exceed 1e12")
        return value

    return {"result": visit(tree.body)}


TOOL_FUNCTIONS = {"lookup_order": lookup_order, "calculator": calculator}
TOOL_SCHEMAS = [
    {"name": "lookup_order", "description": "Look up the current order status.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"], "additionalProperties": False}},
    {"name": "calculator", "description": "Calculate bounded basic arithmetic.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                      "required": ["expression"], "additionalProperties": False}},
]
