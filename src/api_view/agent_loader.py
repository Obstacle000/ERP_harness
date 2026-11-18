"""
AgentLoader 单例
持有 agent 实例、MongoDB 连接、create_config、save/get_display_messages
"""
import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

from ..agent.log_utils import web_logger
from ..agent.schema import ProcurementContext
from .web_config import get_db


@dataclass
class AgentSession:
    """一个用户独享的 Agent、稳定沙箱 Proxy 与串行运行锁。"""

    user_id: str
    username: str
    agent: Any
    # 你看到的 _sessions_lock / _session_build_locks 是「创建/获取 Agent 的结构锁」，而 run_lock 是「Agent 运行时执行对话的业务锁」。
    run_lock: asyncio.Lock


class AgentLoader:
    """Agent 加载器单例 - 管理 Agent 生命周期."""
    # 会给每一个用户配备一个agent实例
    # 为什么AgentLoader需要有协程锁?
    # 我们先看之前做的项目:
    #   1. Agent 图在服务启动时就一次性 compile() 好了，所有用户共用同一个只读图结构，
    #   运行时完全靠 thread_id 走外部数据库隔离，根本不需要在请求里动态建 Agent。
    #   且每一次请求都调用grahph.invoke(),执行图流程
    #   而这里是每个用户配备一个agent,资源共享(mongo_store,client,MongoDBSaver(checkpointer))
    #   每个agent初始化用户专属的 Docker 沙箱.组装全量工具箱（MCP + 业务工具 + 技能读取器）,
    #   配置文件系统三层后端路由（CompositeBackend）将 Agent 看到的文件路径分流到不同存储
    #   加载子 Agent 智能体团队,挂载 14+ 层企业级中间件栈,渲染动态系统提示词,挂载持久化并编译出可执行图
    #
    #   2. 即使每次请求现 new 一个 Agent，它只是组装几个提示词和函数指针，
    #   耗时极短（不到 1 毫秒），且不涉及底层硬件或容器资源(这些在lifespan搞的)。
    # 而在当前这个项目的 AgentLoader 里，协程锁是绝对不能省的，核心原因在于：
    # 它的构建过程包含耗时数秒的 Docker 容器创建(长时间的io同步阻塞)，且发生了协程让出。
    # 比如看get_session()里的agent = await asyncio.to_thread(self._build_agent_for_user, user_id, username)
    # _build_agent_for_user 内部分配 Docker 容器、通过 tar 流把整个项目代码打包上传到沙箱、并在容器内执行一堆初始化 shell 命令。这整个过程需要 2 ~ 5 秒。
    # 如果没有协程锁,比如会发现下面的事情:
    # 请求 A 到达：查字典 self._sessions 发现没有，开始调 Docker 建容器，遇到 await，协程切出。
    # 同一个用户请求 B 同时到达：此时请求 A 的容器还没建好，字典里依旧是空的！请求 B 也开始去建容器。(冲突报错)
    # AgentLoader 的锁设计非常讲究,用的是双重检查锁 DCL
    # 第一个session锁是全局锁,快速检查有没有创建session,有直接返回
    # 没有就创建build_lock用户独立锁,里面再进行一次session存不存在检查,释放锁的时候也在session锁里完成
    # 这么做是因为有可能请求A刚创建Agent,请求B此时拿到了(不是指拿到锁对象,而是进入锁)用户独立锁(之前请求A占着用户独立锁B拿不到),就会导致再创建一次agent
    # 通过双重检查保证了同一时刻只有一个协程在创建资源,创建agent,且创建资源后不会再创建相同agent
    # 用户独立锁对象是存放在字典 self._session_build_locks 里的
    # 如果只有用户独立锁会引发一个及其隐蔽的问题:
    # 请求A进入拿独立锁在创建agent遇到await,挂起,请求B判断完了没有session,打算执行下一行的setdefault
    # 此时请求A创建了agent,所以pop了锁,把锁从字典里删掉了,结束
    # 请求B发现没锁,往里添加了一把,并拿到独立锁,然后下面判断已经又session了,直接退出,锁还没删
    # 导致字典里有一个幽灵锁,谁都删不了他,如果大量用户都发生这种情况,会导致内存泄漏
    # 有了会话锁,把查询session和拿独立锁变成一个原子性操作,所以请求A在pop的时候,请求B查询不了
    # 请求Apop完后,请求B就直接判断存在session走人了

    _instance: Optional['AgentLoader'] = None

    def __new__(cls):
        # 单例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # ``agent`` 保留为旧代码的兼容字段；新链路一律使用 ``get_session``。
        self.agent = None
        self._checkpointer = None
        self._store = None
        self._mongo_client = None
        self._sessions: dict[str, AgentSession] = {}
        # 资源:MongoDB Store / Checkpointer / MongoClient
        # 会话:AgentSession / 线程池创建Agent
        # 这两个锁都不是用户独立的,是全局锁
        self._resources_lock: asyncio.Lock | None = None
        self._sessions_lock: asyncio.Lock | None = None
        self._session_build_locks: dict[str, asyncio.Lock] = {}
        web_logger.info("AgentLoader initialized")

    async def _ensure_resources(self):
        """初始化所有用户共享的 MongoDB Store / Checkpointer。"""
        # 这里也是双重检查,只是资源是共享的,不需要用户隔离
        # 假设请求A遇到await,请求B进来判断还没有,往下走,没锁进不去,请求A执行完毕,释放锁
        # 请求B发现来晚一步已经创建资源,皆大欢喜
        if self._store is not None:
            return

        if self._resources_lock is None:
            self._resources_lock = asyncio.Lock()

        async with self._resources_lock:
            if self._store is not None:
                return
            # 防止阻塞事件循环,await单开一个线程
            await asyncio.to_thread(self._initialize_resources_sync)

    def _initialize_resources_sync(self):
        """阻塞式资源初始化，调用方须放在线程池中。"""
        web_logger.info("Initializing shared Agent resources...")
        try:
            from langgraph.checkpoint.mongodb import MongoDBSaver
            from .mongodb_store import MongoDBStore
            from .web_config import MONGODB_URI, MONGODB_DB_NAME

            # 生产级存储：MongoDB 持久化
            from pymongo import MongoClient
            mongo_client = MongoClient(MONGODB_URI)
            self._checkpointer = MongoDBSaver(mongo_client)
            self._mongo_client = mongo_client
            self._store = MongoDBStore(
                uri=MONGODB_URI,
                db_name=MONGODB_DB_NAME,
                collection_name="langgraph_store",
            )
            web_logger.info("Shared Agent resources initialized")
        except Exception as e:
            web_logger.error(f"Failed to initialize shared Agent resources: {e}")
            raise

    async def initialize(self, user_id: str | None = None, username: str = "用户"):
        """兼容旧调用：初始化共享资源，可选地预创建一个用户会话。"""
        await self._ensure_resources()
        if user_id is not None:
            await self.get_session(user_id, username)

    async def get_session(self, user_id: str, username: str = "用户") -> AgentSession:
        """获取用户独享的 Agent 会话，避免不同用户共享 Sandbox/中间件状态。"""
        await self._ensure_resources()

        if self._sessions_lock is None:
            self._sessions_lock = asyncio.Lock()
        async with self._sessions_lock:
            # _sessions_lock 根本不隔离用户，它是全系统所有用户共享的一把「全局大锁」
            existing = self._sessions.get(user_id)
            if existing is not None:
                return existing
            # 1. 只有当前用户没有 session 时，才去拿这个用户的独立锁
            # 下面只是拿到了锁对象,并不是拿到进入权
            build_lock = self._session_build_locks.setdefault(user_id, asyncio.Lock())

        async with build_lock:
            async with self._sessions_lock:
                # 2. 二次检查，此时有可能前一个并发请求刚刚已经建好了，直接返回
                existing = self._sessions.get(user_id)
                if existing is not None:
                    return existing
            # 3. 只有一个请求会真正进入耗时的 Docker 构建
            agent = await asyncio.to_thread(
                self._build_agent_for_user, user_id, username
            )
            session = AgentSession(
                user_id=user_id,
                username=username,
                agent=agent,
                run_lock=asyncio.Lock(),
            )
            async with self._sessions_lock:
                # 根据userid隔离session
                self._sessions[user_id] = session
                self._session_build_locks.pop(user_id, None)
                # 兼容仍读取该字段的调试脚本。
                if user_id == "default_user":
                    self.agent = agent
            web_logger.info(f"Agent session ready for user: {user_id}")
            return session

    def _build_agent_for_user(self, user_id: str, username: str):
        """在线程池中构建 Agent；其中包含 Docker/Mongo 同步 I/O。"""
        from ..agent.main_agent import create_main_agent

        context = ProcurementContext(user_id=user_id, username=username)
        return create_main_agent(
            user_context=context,
            checkpointer=self._checkpointer,
            store=self._store,
        )

    @property
    def store(self):
        """供请求级工具上下文读取的共享 Store。"""
        return self._store

    def create_config(self, thread_id: str, user_id: str = "default_user") -> dict:
        """创建 LangGraph 运行配置"""
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

    def generate_thread_id(self) -> str:
        """生成新的会话线程ID"""
        return str(uuid.uuid4())

    async def save_display_messages(self, thread_id: str, messages: list):
        """保存前端展示消息到 MongoDB"""
        db = get_db()
        await db.display_messages.update_one(
            {"thread_id": thread_id},
            {"$set": {"messages": messages, "updated_at": datetime.now().isoformat()}},
            upsert=True,
        )

    async def get_display_messages(self, thread_id: str) -> list:
        """获取前端展示消息"""
        db = get_db()
        doc = await db.display_messages.find_one({"thread_id": thread_id})
        return doc.get("messages", []) if doc else []

    async def save_conversation(self, thread_id: str, user_id: str, title: str = "新对话"):
        """保存/更新会话记录"""
        db = get_db()
        await db.conversations.update_one(
            {"thread_id": thread_id},
            {"$set": {"user_id": user_id, "title": title, "updated_at": datetime.now().isoformat()},
             "$setOnInsert": {"created_at": datetime.now().isoformat()}},
            upsert=True,
        )

    async def get_conversations(self, user_id: str) -> list:
        """获取用户的会话列表"""
        db = get_db()
        cursor = db.conversations.find({"user_id": user_id}).sort("updated_at", -1)
        conversations = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            conversations.append(doc)
        return conversations

    async def get_conversation_user_id(self, thread_id: str) -> str | None:
        """查询会话归属，供 resume/state 选择正确的用户 Agent。"""
        db = get_db()
        doc = await db.conversations.find_one({"thread_id": thread_id}, {"user_id": 1})
        return doc.get("user_id") if doc else None

    async def delete_conversation(self, thread_id: str):
        """删除会话"""
        db = get_db()
        await db.conversations.delete_one({"thread_id": thread_id})
        await db.display_messages.delete_one({"thread_id": thread_id})
        await db.harness_traces.delete_one({"thread_id": thread_id})

    async def save_harness_trace(self, thread_id: str, trace: list):
        """保存 Harness 阶段流转 trace 到 MongoDB（可观测/审计）

        记录 Planning → Executing → Review → Result 各阶段的流转时间线，
        以及评审器的结构化判定结果，支持事后审计"哪一步卡住了"。
        """
        db = get_db()
        await db.harness_traces.update_one(
            {"thread_id": thread_id},
            {"$set": {"trace": trace, "updated_at": datetime.now().isoformat()}},
            upsert=True,
        )


# 全局单例
agent_loader = AgentLoader()
