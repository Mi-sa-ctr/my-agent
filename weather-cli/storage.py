import json
from datetime import datetime


class ResultStorage:
    """把查询结果保存到 JSON 文件"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def save(self, data: dict):
        """追加一条天气记录到文件"""
        record = {"timestamp": datetime.now().isoformat(), **data}
        # 用 with 打开文件，自动关闭
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        """读取所有记录"""
        records = []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass  # 文件不存在就返回空列表
        return records

    def clear(self):
        """清空记录"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("")