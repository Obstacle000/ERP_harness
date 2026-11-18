"""
MCP 工具加载模块
使用 langchain-mcp-adapters 连接 MCP Server（SSE模式），获取所有 ERP 工具

生产级实现：
- 连接重试（指数退避）
- 连接健康检查
- 优雅降级（MCP不可用时返回空工具列表，Agent仍可对话）
"""
import asyncio
import time
from typing import List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import MCP_SERVER_URL, MCP_SSE_URL
from ..log_utils import mcp_logger

# 重试配置
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 秒
RETRY_BACKOFF_FACTOR = 2.0

# 缓存（避免重复连接）
_cached_tools: List[BaseTool] | None = None
_cache_time: float = 0
CACHE_TTL = 300  # 5分钟缓存


async def load_mcp_tools(force_refresh: bool = False) -> List[BaseTool]:
    """
    异步加载 MCP 工具列表（带重试和缓存）.

    Args:
        force_refresh: 强制刷新缓存

    Returns:
        MCP 工具列表，连接失败时返回空列表
    """
    # 基于全局变量的模块级缓存
    global _cached_tools, _cache_time

    # 检查缓存
    if not force_refresh and _cached_tools is not None:
        if time.time() - _cache_time < CACHE_TTL:
            return _cached_tools

    mcp_logger.info(f"Connecting to MCP Server: {MCP_SSE_URL}")

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            client = MultiServerMCPClient(
                {
                    "erp": {
                        "url": MCP_SSE_URL,
                        "transport": "sse",
                    }
                }
            )

            tools = await client.get_tools()

            if tools:
                _cached_tools = tools
                _cache_time = time.time()
                mcp_logger.info(
                    f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}"
                )
                return tools
            else:
                mcp_logger.warning("MCP Server returned 0 tools")
                return []

        except Exception as e:
            last_error = e
            delay = RETRY_BASE_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
            mcp_logger.warning(
                f"MCP connection attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)

    mcp_logger.error(
        f"All {MAX_RETRIES} MCP connection attempts failed. Last error: {last_error}. "
        f"Agent will operate without ERP tools."
    )
    return []

# 首先函数改成协程一定就是有利无弊吗?
# 我写成协程,里面同步的部分让他阻塞不就好了,能优化一点是一点?
# create_main_agent是用来组装工具,沙箱,中间件,模型,提示词的车间,里面调用create_deep_agent
# 首先create_main_agent里面有非常多的外部第三方工具,这些工具百分之八十都是同步操作
# 如果create_main_agent改成协程,那么FastAPI理所应当会什么都不管直接用await调用
# 但是他不知道里面有非常多的同步阻塞,这些同步工具会阻塞事件循环,假设卡死了1秒钟
# 这个期间内所有的http请求都在排队,有其他用户的 SSE 打字机流式输出瞬间全部卡顿停滞
# 所以我们应该显示告诉FastAPI把create_main_agent写成函数,让他用await asyncio.to_thread(create_main_agent)
# 把一个普通的同步阻塞函数，扔到后台的另一个线程里去执行，从而避免卡死当前的事件循环
# 事件循环里跑的协程，其内部每一步要么是 await 真正的异步 I/O，要么是极短的 CPU 计算。一旦出现长同步阻塞（哪怕 100ms 级、在高并发下），就得挪到线程/进程。
# 所以create_main_agent是同步函数,这就导致里面用不了await,所以需要一个同步包装
def load_mcp_tools_sync(force_refresh: bool = False) -> List[BaseTool]:
    """
    同步包装：加载 MCP 工具列表(不是数据同步的意思)

    处理各种事件循环状态：
    - 无事件循环 → asyncio.run()
    - 有运行中的事件循环 → ThreadPoolExecutor + asyncio.run()
    """
    # 根据当前代码运行在“无事件循环”还是“已有事件循环（FastAPI 运行中）”，做到了把协程转成同步阻塞返回
    # 当前load_mcp_tools_sync被调用在两个不同的上下文里
    # 1. 同步测试环境(函数内调用) 2. 异步测试环境(协程内调用)
    # 这个函数写成这样，在工业界叫做 “自适应双模包装器（Polymorphic Runner / Universal Wrapper）”：
    # 作者希望 load_mcp_tools_sync() 是一个万能工具函数

    # 先检查缓存
    global _cached_tools, _cache_time
    if not force_refresh and _cached_tools is not None:
        if time.time() - _cache_time < CACHE_TTL:
            return _cached_tools

    try:
        # 1. 检测当前线程有没有正在跑的事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 场景 A：当前已经在异步环境（如 FastAPI 主线程中）
        # 如果是在 FastAPI 运行中，主线程里已经有一个现成的事件循环在跑了,
        # 此时如果你直接调 asyncio.run()，Python 会直接报致命错误:
        # 禁止在同一线程嵌套启动事件循环
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            # 1. 派单给一个全新的子线程
            future = pool.submit(asyncio.run, load_mcp_tools(force_refresh))
            # 2. 主线程在这里同步死等结果
            # future.result() 会阻塞调用它的线程
            return future.result(timeout=30)
    else:
        # 场景 B：当前在普通同步脚本/测试环境（没有事件循环）
        # asyncio.run() 是 Python 官方提供的标准桥梁。
        # 它在当前线程临时创建一个全新的事件循环，把你的协程 load_mcp_tools() 扔进去跑，当前线程一直阻塞等待，
        # 直到协程跑完并 return 结果，然后它自动把这个临时事件循环关闭销毁。
        return asyncio.run(load_mcp_tools(force_refresh))


def invalidate_cache():
    """清除工具缓存（MCP Server 重启后调用）"""
    global _cached_tools, _cache_time
    _cached_tools = None
    _cache_time = 0
    mcp_logger.info("MCP tools cache invalidated")
