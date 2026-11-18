"""
HTTP 基础客户端模块
httpx AsyncClient 单例（连接池、超时），封装 GET/POST/PUT/PATCH/DELETE
"""
import httpx
import json
from typing import Any, Optional
from .server_config import ERP_BASE_URL, HTTP_TIMEOUT, HTTP_MAX_CONNECTIONS


class ERPHttpClient:
    """ERP 后端 HTTP 客户端单例"""
    """
    HTTP 客户端采用单例模式，因为其维护昂贵的连接池资源，全局共享可复用 TCP 连接、
    显著提升性能并统一超时/并发配置；而 LLM 实例不适合单例，因为不同任务需要独立的模型、温度、API Key 等参数配置，
    且调用本身是无状态的，按需创建更灵活、更经济。
    """
    _instance: Optional['ERPHttpClient'] = None
    _client: Optional[httpx.AsyncClient] = None

    # 通过 重写 Python 类的 __new__ 方法 来实现单例模式
    # 负责创建并返回一个类的实例。
    # 虽然平时用得最多的是 __init__，但其实在 Python 里，创建对象的过程是分两步走的：
    #   1. __new__（造物主）：负责在内存中凭空捏造出一个对象，并返回它。
    #   2. __init__（装修工）：负责接收这个刚造出来的对象，给它初始化（比如赋初始值）。
    def __new__(cls):
        if cls._instance is None:
            # 只有第一次调用时，才真正去创建对象
            cls._instance = super().__new__(cls)
        # 无论后面调用多少次，都直接返回第一次创建好的那个对象
        return cls._instance

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=ERP_BASE_URL,
                timeout=httpx.Timeout(HTTP_TIMEOUT),
                limits=httpx.Limits(
                    max_connections=HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=10,
                ),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET 请求"""
        # 过滤掉 None 值的参数
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        response = await self.client.get(path, params=params)
        return self._handle_response(response)

    async def post(self, path: str, json_data: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        """POST 请求"""
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        response = await self.client.post(path, json=json_data, params=params)
        return self._handle_response(response)

    async def put(self, path: str, json_data: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        """PUT 请求"""
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        response = await self.client.put(path, json=json_data, params=params)
        return self._handle_response(response)

    async def patch(self, path: str, params: Optional[dict] = None) -> dict:
        """PATCH 请求"""
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        response = await self.client.patch(path, params=params)
        return self._handle_response(response)

    async def delete(self, path: str) -> dict:
        """DELETE 请求"""
        response = await self.client.delete(path)
        return self._handle_response(response)

    def _handle_response(self, response: httpx.Response) -> dict:
        """统一响应处理"""
        try:
            data = response.json()
            if response.status_code == 200:
                return data
            else:
                return {
                    "code": response.status_code,
                    "message": f"HTTP Error: {response.status_code}",
                    "data": data,
                }
        except json.JSONDecodeError:
            return {
                "code": response.status_code,
                "message": f"Response not JSON: {response.text[:500]}",
                "data": None,
            }

    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 全局单例
erp_client = ERPHttpClient()
