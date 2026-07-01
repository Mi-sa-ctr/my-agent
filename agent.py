from openai import OpenAI
import json

client = OpenAI(api_key="sk-94dcdadb6dd44885921fba10626447d9", base_url="https://api.deepseek.com")


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

#计算器函数这里没看啊def

tool1 = {"type": "function","function": {"name": "calculator","description": "计算数学表达式，可用+ - * / （）等符号" },"parameters":{"type": "object","properties": {"expression": {"type": "string","description":"数学表达式，比如\"3+3=6\""}}},"required":["expression"]}
tools = [tool1]
tool_map={"calculator":calculator}

messages=[{"role":"user","content":"计算 (23 + 17) * 4 的结果"}]
response = client.chat.completions.create(model="deepseek-chat",messages= messages,tools=tools,max_tokens=1024)
print(f"第1轮 finish_reason: {response.choices[0].finish_reason}")


if response.choices[0].finish_reason=="tool_calls":
    messages.append(response.choices[0].message.model_dump())

    for tc in response.choices[0].message.tool_calls:
        print(f"模型想要调用,现在执行的工具为：{tc.function.name}")

        arg=json.loads(tc.function.arguments)
        result=tool_map[tc.function.name](**arg)
        print(f"结果: {result}")

        messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
    
    response2=client.chat.completions.create(model="deepseek-chat",messages= messages,tools=tools,max_tokens=1024)
    print(f"\n第2轮 finish_reason: {response2.choices[0].finish_reason}")
    print(f"答案是：{response2.choices[0].message.content}")


