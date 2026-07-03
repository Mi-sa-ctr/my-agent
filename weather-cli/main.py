from api_client import WeatherClientWithCache
from storage import ResultStorage

def main():
    # 创建客户端对象
    client = WeatherClientWithCache(timeout=15, max_cache=100)
    storage = ResultStorage("weather_history.jsonl")

    try:
        # 查几个城市
        cities = [
            ("北京", 39.9, 116.4),
            ("上海", 31.2, 121.5),
            ("深圳", 22.5, 114.1),
        ]
        for name, lat, lon in cities:
            print(f"查询 {name}...")
            result = client.get_weather(lat, lon)
            result["city"] = name               # 用真实城市名覆盖
            print(f"  🌡 {result['temperature']}°C, 风速 {result['windspeed']}km/h")
            storage.save(result)

        # 用 property 查看缓存状态
        print(f"\n缓存中有 {client.cache_size} 条记录")

        # 再查一次北京 → 应该命中缓存
        print("\n再次查询北京（应命中缓存）...")
        client.get_weather(39.9, 116.4)

    except (ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"❌ {e}")

    # 查看历史记录
    print(f"\n历史记录（{storage.filepath}）：")
    for r in storage.load_all():
        print(f"  {r['timestamp'][:19]} | {r['city']}: {r['temperature']}°C")

if __name__ == "__main__":
    main()