import requests
import json


class WeatherAPIClient:
    """基础天气 API 客户端"""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 10):
        """
        初始化客户端
        :param timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        self._last_response = None  # 私有属性，存上次请求结果

    # ===== property 装饰器：把方法当属性用 =====
    @property
    def last_city(self) -> str | None:
        """上次查询的城市名"""
        return self._last_response.get("city") if self._last_response else None

    @property
    def last_temperature(self) -> float | None:
        """上次查询的温度"""
        return self._last_response.get("temperature") if self._last_response else None

    # ===== 核心方法 =====
    def get_weather(self, latitude: float, longitude: float) -> dict:
        """
        获取天气，自动处理异常
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "timezone": "auto",
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()  # 非 200 自动抛异常
            data = response.json()
            with open("weathertest001.json","w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            result = {
                "city": f"{latitude},{longitude}",
                "temperature": data["current_weather"]["temperature"],
                "windspeed": data["current_weather"]["windspeed"],
                "weathercode": data["current_weather"]["weathercode"],
            }
            self._last_response = result
            return result
        except requests.ConnectionError:
            raise ConnectionError("网络连接失败，请检查网络")
        except requests.Timeout:
            raise TimeoutError(f"请求超时（{self.timeout}秒），请稍后重试")
        except requests.HTTPError as e:
            raise RuntimeError(f"API 返回错误: {e}")

    def __repr__(self):
        return f"WeatherAPIClient(timeout={self.timeout})"


# ===== 继承：扩展功能 =====
class WeatherClientWithCache(WeatherAPIClient):
    """带缓存的天气客户端 —— 继承基类"""

    def __init__(self, timeout: int = 10, max_cache: int = 50):
        """
        初始化，扩展了父类的 __init__
        :param max_cache: 最大缓存数量
        """
        super().__init__(timeout)           # 调父类 __init__
        self._cache = {}                    # {坐标: 结果}
        self.max_cache = max_cache

    def get_weather(self, latitude: float, longitude: float) -> dict:
        """重写父类方法：先查缓存，没有再调 API"""
        key = f"{latitude},{longitude}"
        if key in self._cache:
            print(f"[缓存命中] {key}")
            return self._cache[key]
        result = super().get_weather(latitude, longitude)  # 调父类方法
        if len(self._cache) >= self.max_cache:
            self._cache.pop(next(iter(self._cache)))       # 删最早一条
        self._cache[key] = result
        return result

    @property
    def cache_size(self) -> int:
        return len(self._cache)