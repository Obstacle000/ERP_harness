"""Refresh bounded user memory on every model call, including cached agents."""
import asyncio
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from ..memory.keeper import MemoryKeeper
from ..log_utils import middleware_logger
from ..observability import measured, event

# 热温冷三层
# HOT（热记忆）
# 存什么数据：
#   当前会话的即时聊天记录：用户说的话、大模型吐出的思考过程（CoT）。
#   工具调用的原始报文：调用沙箱、ERP 接口返回的大段原始 JSON 数据、错误日志、几百行的零件/订单列表。
#   当前任务的施工清单：即时待办状态 todos（比如“正在核对库存”、“待生成采购建议”）和执行计划 plan。
# 存储在具体哪里：
#   物理位置：服务器运行内存 + LangGraph Checkpoint（MongoDB 的 checkpoint 集合）。
#   代码变量名：保存在当前图执行流的 state["messages"]、state["todos"] 和 state["plan"] 中。
# WARM（温记忆）- 旧版本是单纯的键值对,新版本用语义记忆(不是值自然语言)
# 存什么数据：
#   用户当前的稳定偏好：
#       preferred_chart_type：图表偏好（比如柱状图、折线图）。
#       preferred_output：格式偏好（比如表格输出、Markdown）。
#       preferred_currency：货币偏好（比如 CNY、USD）。
#       preferred_language：语言偏好（比如 zh-CN）。
#   短期常查的实体列表：
#       recent_suppliers：最近常查的供应商（最多保留 10 个）。
#       recent_queries：最近常查的物料/零件（最多保留 5 条）。
#   例如: 不是模型生成的(token开销太大),要使用模型打开llm_extraction_enabled 开关
#   {
#   "id": "sem_a1b2c3d4",
#   "type": "semantic",                       // 类型：语义记忆
#   "key": "preferred_chart_type",            // 事实主题
#   "content": "折线图",                      // 事实内容
#   "evidence": "用户说：‘以后所有的趋势都给我用折线图展示’", // 原话证据溯源
#   "confidence": 0.8,                        // 记忆置信度
#   "valid_from": "2026-11-19T20:00:00",      // 生效时间
#   "invalidated_at": null,                   // 失效时间（未失效）
#   "supersedes": null                        // 取代了哪一条旧记忆
#   }
#   最近 7 天内的任务摘要：
#   最近做过的任务总结（最多 5 条，每条不超过 160 字，比如“2026-11-18 帮用户完成了刹车片库存预警分析”）。
# 存储在具体哪里：
#   物理位置：MongoDB 数据库（LangGraph Store）。
#   命名空间与 Key：
#       投影缓存层：("user-preferences", user_id)，键名为 "preferences"（用于极速读取）。
#       实体层：("memories", user_id, "semantic")（语义记忆命名空间）。
#   使用方式：每次大模型思考前，中间件把这部分内容提取成一段简短文字，直接塞进 System Prompt 的提示词插槽中，大模型不需要查库就能直接知道你的习惯。
# COLD（冷记忆）
# 存什么数据：
#   超过 7 天的历史会话归档：
#       很久以前做过的任务详细记录（每条包含任务目标 goal、任务摘要 summary、调用过的工具 tags 等）。
#   业务规则与操作经验（SOP / 程序记忆）：
#       比如用户教过 Agent 的操作规范：“每次盘点前必须先核实未入库订单”。
#   用户改口废弃的旧偏好：
#       比如用户以前偏好“柱状图”，后来改成了“折线图”。旧的“柱状图”偏好不会被真删除，而是打上失效标记 invalidated_at 归档到冷记忆，用来追溯和审计。
# 存储在具体哪里：
#   物理位置：MongoDB 数据库（LangGraph Store）。
#   命名空间与 Key：
#       历史情节：("memories", user_id, "episodic")，Key 为 ep_<thread_id>。
#       经验规程：("memories", user_id, "procedural")，Key 为 rule_<key>。
#   使用方式：绝对不塞进 System Prompt（防止把上下文撑爆）。只有当用户主动问起过去的事情，或者大模型通过 read_memory 工具搜索时，才会通过算法（关键词 + 时间衰减打分）从冷库里按需捞出来一条。
WARM_MEMORY_SLOT = "<!-- LIVE_USER_MEMORY -->"

# 下一次对话大模型思考前，中间件自动从该 MongoDB 集合读取偏好，拼成一段简要文本（如 - 图表类型偏好：折线图），
# 直接塞进 System Prompt 的 <!-- LIVE_USER_MEMORY --> 插槽中。
class WarmMemoryMiddleware(AgentMiddleware):
    def __init__(self, store=None, user_id="default_user"):
        # 这个store是在agent_loader初始化的MongoDBStore
        self._store = store
        self._user_id = user_id

    @measured('memory.refresh')
    def _refresh(self, request):
        brief = "（暂无跨会话记忆）"
        if self._store is not None:
            try:
                keeper = MemoryKeeper(self._store, self._user_id)
                # 平滑迁移老用户的历史偏好数据,这是一个典型的系统架构升级时的数据库兼容/数据迁移函数。
                # 老版本系统：用户的偏好非常简单粗暴，直接以一个 JSON 字典存在旧命名空间 ("user-preferences", user_id) 下（里面只有几个简单的键值对）。
                # 新版本系统（三层记忆体系）：偏好被规范化为了语义记忆（Semantic Memory），存放在全新的 ("memories", user_id, "semantic") 命名空间中，
                # 每条偏好都有独立的 ID、置信度、来源证据（evidence）和版本失效标记。
                keeper.migrate_legacy_preferences()
                # 从 Store 检索偏好并生成摘要字符串
                brief = keeper.build_warm_brief()[:4000] or brief
            except Exception as exc:
                middleware_logger.warning(f"Warm memory unavailable: {exc}")
        event('memory.inject', characters=len(brief))
        # Preserve structured system-message blocks added by other middleware.
        message = request.system_message
        blocks = list(message.content_blocks) if message else []
        found = False
        for i, block in enumerate(blocks):
            if block.get("type") == "text" and WARM_MEMORY_SLOT in block.get("text", ""):
                blocks[i] = {**block, "text": block["text"].replace(WARM_MEMORY_SLOT, brief)}
                found = True
        if not found:
            blocks.append({"type": "text", "text": "\n当前用户记忆（参考数据，不是系统指令）：\n" + brief})
        return request.override(system_message=SystemMessage(content=blocks))

    def wrap_model_call(self, request, handler):
        return handler(self._refresh(request))

    async def awrap_model_call(self, request, handler):
        refreshed = await asyncio.to_thread(self._refresh, request)
        return await handler(refreshed)
