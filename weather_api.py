import requests
import json

city = "上海"

# 方法1：JSON 格式获取（适合程序解析）
url = f"https://wttr.in/{city}?format=j1"
resp = requests.get(url)
data = resp.json()

with open("weather.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 提取当前天气
current = data["current_condition"][0]
temp = current["temp_C"]
humidity = current["humidity"]
weather_desc = current["weatherDesc"][0]["value"]

print(f"📍 {city} 当前天气:")
print(f"   温度: {temp}°C")
print(f"   湿度: {humidity}%")
print(f"   天气: {weather_desc}")

# 方法2：纯文本格式（直接给人看）
print("\n--- 未来3天预报 ---")
url2 = f"https://wttr.in/{city}?format=3"
resp2 = requests.get(url2)
print(resp2.text)