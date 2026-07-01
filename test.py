from openai import OpenAI
import json

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com",
)

# 定义一个真正可用的计算器工具
def calculator(expression: str) -> str:
    """安全的计算器，只支持基本运算"""
    allowed = set("0123456789+-*/().% ^")
    if not all(c in allowed for c in expression):
        return f"错误：表达式包含不允许的字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 工具定义 + 执行函数映射
tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式，支持 + - * / 和括号",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    }
}]

tool_map = {"calculator": calculator}

messages = [{"role": "user", "content": "计算 (23 + 17) * 4 的结果"}]

# 第一轮：DeepSeek 决定是否调用工具
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    max_tokens=1024,
)

msg = response.choices[0].message
print(f"第1轮 finish_reason: {response.choices[0].finish_reason}")

if response.choices[0].finish_reason == "tool_calls":
    # 把模型的回复加入消息历史
    messages.append(msg.model_dump())
    
    # 执行每个工具调用
    for tc in msg.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        print(f"执行工具: {func_name}({func_args})")
        result = tool_map[func_name](**func_args)
        print(f"结果: {result}")
        
        # 把工具结果作为 tool 消息追加
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    
    # 第二轮：DeepSeek 基于工具结果生成最终回答
    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
    )
    print(f"\n第2轮 finish_reason: {response2.choices[0].finish_reason}")
    print(f"最终回答: {response2.choices[0].message.content}")