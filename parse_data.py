import json

# 读取 JSON 文件
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 访问数据
print(f"姓名: {data['name']}")
print(f"技能: {', '.join(data['skills'])}")
print("项目列表:")
for p in data["projects"]:
    print(f"  - {p['title']} ({p['status']})")

# 练习：把数据改一下再写回去
data["skills"].append("LangChain")
with open("data_new.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n✅ 新文件已写入 data_new.json")