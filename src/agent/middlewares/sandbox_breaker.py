"""
中间件 7: 沙箱熔断器
连续 N 次沙箱工具调用失败 → 短路 → 降级响应（避免无限重试拖垮系统）。
"""
import time
from typing import Any, Callable, Awaitable

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents.middleware import AgentMiddleware, Runtime, ToolCallRequest
from ..log_utils import middleware_logger


class SandboxCircuitBreakerMiddleware(AgentMiddleware):
    """
    沙箱熔断器中间件

    三态模型：
    - CLOSED（正常）：所有工具调用正常通过
    - OPEN（熔断）：沙箱相关工具直接返回降级响应，不调用 handler
    - HALF_OPEN（探测）：熔断后经过 recovery_timeout，允许一次试探

    触发条件：连续 failure_threshold 次沙箱工具调用失败
    恢复条件：HALF_OPEN 状态下一次成功调用 → 回到 CLOSED

    仅拦截沙箱相关工具（execute, run_code 等），MCP/ERP 工具不受影响。
    """

    # 沙箱相关工具名（这些工具依赖沙箱容器）
    SANDBOX_TOOLS = {"execute", "run_code", "run_python", "shell", "bash"}

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._state: str = "CLOSED"  # CLOSED / OPEN / HALF_OPEN
        self._last_failure_time: float = 0
        self.tools = []

    @property
    def name(self) -> str:
        return "SandboxCircuitBreakerMiddleware"

    def _is_sandbox_tool(self, tool_name: str) -> bool:
        """判断工具是否依赖沙箱"""
        return tool_name.lower() in self.SANDBOX_TOOLS
    # 这个叫做工具环绕拦截钩子,他包裹的是工具调用
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """拦截沙箱工具调用，实现熔断逻辑"""
        # 熔断器:
        #         工具调用成功
        #      ┌─────────────────┐
        #      │                 ▼
        # ┌───────────┐    连续失败 3 次      ┌───────────┐
        # │  CLOSED   │ ─────────────────>  │   OPEN    │
        # │  (放行)   │                      │ (内存短路) │
        # └───────────┘                     └───────────┘
        #       ▲                                │
        #       │ 试探成功                         │ 冷却等待 60 秒后
        #       │                                ▼
        #       │                          ┌───────────┐
        #       └───────────────────────── │ HALF_OPEN │
        #                试探失败            │ (放行一次) │
        #                (重置60秒)         └───────────┘
        # 初始状态。所有 execute / run_code 请求正常打进沙箱,只要调用成功，_failure_count 立即清零。
        # 第一次调用失败,进入_on_failure,_failure_count+1,记录失败日志,模型不管继续下一轮
        # 第三次调用失败,进入_on_failure,c>=3,开启熔断,状态改为"OPEN",记录日志,下一轮
        # 判断状态为OPEN,判断倒计时到没到,没到就返回ToolMessage,放进message,让模型判断下一步
        # 到了就把状态改成"HALF_OPEN",试探性调用handler(request),调用工具,如果成功调用_on_success,关闭熔断
        # 失败调用_on_failure,返回ToolMessage,模型判断继续等还是直接输出
        # 按理来说同一个模型不能熬过60秒的等待时间,在这之前模型已经输出了,所以60秒是在防新来的模型
        tool_name = request.tool_call.get("name", "")

        # 非沙箱工具直接通过
        if not self._is_sandbox_tool(tool_name):
            return await handler(request)

        # 熔断器状态判断
        if self._state == "OPEN":
            # 检查是否到达恢复时间
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = "HALF_OPEN"
                middleware_logger.info("Circuit breaker → HALF_OPEN (attempting recovery)")
            else:
                # 直接返回降级响应
                middleware_logger.warning(
                    f"Circuit breaker OPEN: skipping sandbox tool '{tool_name}'"
                )
                # 这个 ToolMessage 被追加进当前会话的 messages 列表,进入下一轮,大模型就能看到,决定下一步干嘛
                return ToolMessage(
                    content=f"[沙箱暂时不可用] 工具 '{tool_name}' 被熔断器拦截。"
                            f"沙箱服务正在恢复中，请稍后重试或使用其他工具完成任务。",
                    tool_call_id=request.tool_call.get("id", ""),
                    status="error",
                )


        # CLOSED 或 HALF_OPEN：执行工具
        try:
            # # 1. 真实放行调用沙箱工具
            result = await handler(request)
            # 成功 → 重置
            self.c()
            return result
        except Exception as e:
            self._on_failure(tool_name, e)# 失败计数 +1
            # 返回错误消息而非抛出异常（让 Agent 知道失败了但可以继续）
            return ToolMessage(
                content=f"[沙箱调用失败] {tool_name}: {str(e)[:200]}。"
                        f"连续失败 {self._failure_count} 次。",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

    def _on_success(self):
        """工具调用成功"""
        if self._state == "HALF_OPEN":
            middleware_logger.info("Circuit breaker → CLOSED (sandbox recovered)")
        self._failure_count = 0
        self._state = "CLOSED"

    def _on_failure(self, tool_name: str, error: Exception):
        """工具调用失败"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            self._state = "OPEN"
            middleware_logger.error(
                f"Circuit breaker → OPEN after {self._failure_count} failures. "
                f"Last error: {tool_name}: {error}"
            )
        else:
            middleware_logger.warning(
                f"Sandbox tool failure ({self._failure_count}/{self._failure_threshold}): "
                f"{tool_name}: {error}"
            )
