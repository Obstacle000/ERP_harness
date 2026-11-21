"""
主入口：create_main_agent() + precompute_agent_context()
DeepAgent 核心组装 — 严格遵循 Harness Engineering 架构
"""
import io
import tarfile
import fnmatch
from typing import Optional
from pathlib import Path

from deepagents import create_deep_agent, RubricMiddleware
from deepagents.backends import CompositeBackend, StoreBackend, LocalShellBackend
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from .config import (
    get_llm,
    INTERRUPT_ON_TOOLS, skills_store_namespace,
    MAX_MODEL_CALLS, MAX_TOOL_CALLS,
    SANDBOX_HEALTH_CHECK_INTERVAL_SECONDS,
)
from .schema import ProcurementContext
from .log_utils import agent_logger
from .memory.config import DEFAULT_MEMORY_CONFIG
from .memory.keeper import MemoryKeeper
from .memory.prompts import MAIN_SYSTEM_PROMPT, MEMORY_USAGE_PROMPT

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / "src" / "skills"
MEMORY_DIR = Path(__file__).parent / "memory"


def _upload_project_to_sandbox(sandbox) -> bool:
    """
    将项目文件上传到沙箱 /workspace/src/ 目录，搭建 1:1 开发环境。
    跳过 node_modules / .git / __pycache__ / .next 等无关目录。
    """
    skip_patterns = [
        ".git", "__pycache__", "node_modules", ".next", ".venv",
        ".env", ".env.*", "*.pem", "*.key",
        ".eggs", "*.pyc", "*.pyo", ".DS_Store", ".docker",
        "*.png", "*.jpg", "*.jpeg", "*.gif",  # 生成物不传
    ]

    def _should_skip(rel_path: str) -> bool:
        for pattern in skip_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            parts = rel_path.replace("\\", "/").split("/")
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    remote_dir = "/workspace/src"
    try:
        sandbox.execute(f"mkdir -p '{remote_dir}'")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
            # 把 PROJECT_ROOT 目录下的文件打包成一个 gzip 压缩的 tar 包，写入内存中的 tar_stream。
            # 用 rglob("*") 递归遍历整个项目根目录
            for file_path in sorted(PROJECT_ROOT.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if _should_skip(rel):
                    continue
                # arcname=rel 保证压缩包内是相对路径，解压后不会多出一层 PROJECT_ROOT 的绝对路径。
                tar.add(str(file_path), arcname=rel)
        # 最后 seek(0) 回到开头，read() 出完整字节流 data。
        tar_stream.seek(0)
        data = tar_stream.read()

        # 仅使用 BaseSandbox 的公开协议上传；这样 SandboxBackendProxy 热替换
        # 底层容器后，项目上传仍能工作，且不会依赖 Docker 私有 _container。
        archive_path = "/tmp/erp-agent-project.tar.gz"
        responses = sandbox.upload_files([(archive_path, data)])
        errors = [getattr(item, "error", None) for item in responses]
        if not responses or any(errors):
            raise RuntimeError(f"archive upload failed: {errors or 'no response'}")
        # -C /workspace/src：解压到目标目录。
        # && rm -f：解压成功后立即删除压缩包，不占沙箱空间。
        unpack = sandbox.execute(
            f"tar -xzf '{archive_path}' -C '{remote_dir}' && rm -f '{archive_path}'"
        )
        # 检查 exit_code，失败就抛异常。
        if unpack.exit_code != 0:
            raise RuntimeError(f"archive extraction failed: {unpack.output}")

        # find ... -type f | wc -l：统计解压后文件数
        count_result = sandbox.execute(
            f"find {remote_dir} -type f 2>/dev/null | wc -l"
        )
        # du -sh ... | cut -f1：统计总大小。
        size_result = sandbox.execute(
            f"du -sh {remote_dir} 2>/dev/null | cut -f1"
        )
        file_count = count_result.output.strip() if count_result.exit_code == 0 else "?"
        total_size = size_result.output.strip() if size_result.exit_code == 0 else "?"

        agent_logger.info(
            f"Project uploaded to sandbox: {remote_dir} "
            f"({file_count} files, {total_size})"
        )
        return True
    except Exception as e:
        agent_logger.warning(f"Project upload to sandbox failed (non-fatal): {e}")
        return False


def bootstrap_sandbox(sandbox) -> None:
    """填充一个新领取或重建后的沙箱，使其可立即继续 Agent 工作。

    这个函数只使用稳定 Proxy 暴露的公开能力，因此健康检查重建后的同一个
    Agent、CompositeBackend 和工具对象都能继续复用。
    """
    setup = sandbox.execute(
        "mkdir -p /workspace/charts /workspace/output /workspace/data "
        "/workspace/python-packages /skills"
    )
    if setup.exit_code != 0:
        raise RuntimeError(f"sandbox directory setup failed: {setup.output}")

    if not _upload_project_to_sandbox(sandbox):
        raise RuntimeError("project bootstrap upload failed")

    skills_copy = sandbox.execute(
        "mkdir -p /skills && "
        "cp -r /workspace/src/src/skills/. /skills/ 2>/dev/null || true"
    )
    if skills_copy.exit_code != 0:
        raise RuntimeError(f"core Skills bootstrap failed: {skills_copy.output}")

    agent_logger.info("Sandbox bootstrap complete: project + core Skills restored")


def _load_skills_prompt(skills_dir: Path) -> str:
    """
    兼容入口：仅返回基础技能元数据，不读取正文注入提示词。
    """
    from .progressive_skills import SkillCatalog
    return SkillCatalog(skills_dir).index()


async def precompute_agent_context(
    user_id: str = "default_user",
    username: str = "用户",
    store=None,
) -> ProcurementContext:
    """
    预计算 Agent 上下文

    从 Store 加载用户历史偏好与 WARM 层记忆摘要，构造完整的 ProcurementContext。
    如果 Store 不可用则使用默认值。
    """
    preferences = {}
    warm_memory = ""

    if store is not None:
        try:
            namespace = ("user-preferences", user_id)
            items = store.search(namespace)
            for item in items:
                if item.key == "preferences" and isinstance(item.value, dict):
                    preferences.update(item.value)
            if preferences:
                agent_logger.info(f"Loaded preferences for {user_id}: {list(preferences.keys())}")
        except Exception as e:
            agent_logger.warning(f"Failed to load preferences: {e}")

        # WARM 层：语义记忆（偏好）+ 近期情节摘要，常驻注入系统提示词
        try:
            keeper = MemoryKeeper(store, user_id, DEFAULT_MEMORY_CONFIG)
            keeper.migrate_legacy_preferences()
            warm_memory = keeper.build_warm_brief()
            if not preferences:
                preferences.update(keeper.load_preferences())
        except Exception as e:
            agent_logger.warning(f"Failed to build warm memory: {e}")

    return ProcurementContext(
        user_id=user_id,
        username=username,
        preferences=preferences,
        warm_memory=warm_memory,
    )


def create_main_agent(
    user_context: Optional[ProcurementContext] = None,
    checkpointer=None,
    store=None,
):
    """
    创建主 Agent 实例

    组装顺序（严格遵循文档）：
    1. LLM（DeepSeek）
    2. CompositeBackend 三层路由（default=sandbox, /memories/=Store, /persisted-skills/=Store）
    3. 工具加载（MCP + chart + web_search + hitl）
    4. 子Agent配置（YAML声明式）
    5. 中间件栈（7自定义 + 框架内置）
    6. 创建 Agent

    Args:
        user_context: 用户上下文（含偏好）
        checkpointer: LangGraph checkpointer（会话持久化）
        store: LangGraph store（跨会话存储）

    Returns:
        CompiledStateGraph - 可执行的 Agent 图
    """
    if user_context is None:
        user_context = ProcurementContext()

    agent_logger.info(f"Creating main agent for user: {user_context.user_id}")

    # ===== 1. LLM =====
    llm = get_llm()
    grader_llm = get_llm(thinking=False)

    # ===== 2. Backend - Manager + Stable Proxy + CompositeBackend =====
    # Manager 负责用户级领取/恢复/重建；Proxy 保持 Agent 已持有的后端引用稳定。
    sandbox_manager_instance = None
    managed_sandbox = False
    try:
        from .backends.sandbox_manager import sandbox_manager
        from .backends.sandbox_proxy import SandboxBackendProxy
        from .backends.sandbox_holder import set_sandbox

        raw_sandbox = sandbox_manager.get_sandbox(user_context.user_id)
        sandbox_backend = SandboxBackendProxy(raw_sandbox)
        bootstrap_sandbox(sandbox_backend)
        sandbox_manager_instance = sandbox_manager
        managed_sandbox = True
        # 注册的是稳定 Proxy，而不是会在重建后失效的原始 Docker 连接。
        set_sandbox(sandbox_backend, user_context.user_id)
        agent_logger.info(
            f"Using managed Docker sandbox for {user_context.user_id}: "
            f"{sandbox_backend.container_name}"
        )
    except Exception as e:
        agent_logger.warning(
            f"Managed Docker sandbox unavailable ({e}), falling back to LocalShell"
        )
        sandbox_backend = LocalShellBackend(virtual_mode=True)
        from .backends.sandbox_holder import set_sandbox
        set_sandbox(None, user_context.user_id)

    # deepagents >= 0.7 要求传入已初始化的 Backend 实例，
    # 不再接受旧版的 backend factory。
    backend_routes = {}
    if store is not None:
        backend_routes["/memories/"] = StoreBackend(
            namespace=lambda _rt: (user_context.user_id,),
        )
        backend_routes["/persisted-skills/"] = StoreBackend(
            namespace=lambda _rt: skills_store_namespace(user_context.user_id),
        )
    composite_backend = CompositeBackend(
        default=sandbox_backend,
        routes=backend_routes,
    )

    # ===== 3. 加载工具 =====
    from .tools.mcp_client import load_mcp_tools_sync
    from .tools.chart_generator import generate_chart
    from .tools.web_search import web_search
    from .tools.web_fetch import web_fetch, install_skill
    from .tools.skill_management import list_user_skills
    from .tools.hitl_tools import request_order_info
    from .tools.download_sandbox_file import download_sandbox_file, list_sandbox_files
    from .tools.document_generator import generate_document, generate_table_report

    mcp_tools = load_mcp_tools_sync()
    custom_tools = [generate_chart, web_search, web_fetch, install_skill, list_user_skills, request_order_info,
                    download_sandbox_file, list_sandbox_files,
                    generate_document, generate_table_report]

    # 三层记忆：COLD 层按需检索 + 显式写入（WARM 层已在系统提示词里常驻注入）
    if DEFAULT_MEMORY_CONFIG.memory_tools_enabled:
        from .tools.memory_tools import read_memory, remember
        custom_tools.extend([read_memory, remember])
    all_tools = mcp_tools + custom_tools

    # Prompt disclosure is independent from physical sandbox file synchronization.
    from .progressive_skills import (
        SkillCatalog, ProgressiveSkillsMiddleware, attach_to_subagents, SLOT,
    )
    skill_catalog = SkillCatalog(SKILLS_DIR, store=store, user_id=user_context.user_id,
                                 sandbox=managed_sandbox)
    skill_read_tools = skill_catalog.tools()
    all_tools.extend(skill_read_tools)

    agent_logger.info(
        f"Tools loaded: {len(mcp_tools)} MCP + {len(custom_tools)} custom + "
        f"{len(skill_read_tools)} skill readers = {len(all_tools)} total"
    )

    # ===== 4. 加载子Agent配置 =====
    from .subagents.loader import load_subagent_configs, resolve_subagent_tools, get_delegation_context_prompt
    subagent_configs = load_subagent_configs()
    subagents = resolve_subagent_tools(subagent_configs, all_tools)
    attach_to_subagents(subagents, skill_catalog, skill_read_tools)
    # 生成委派上下文协议（注入主 Agent 提示词）
    delegation_prompt = get_delegation_context_prompt(subagent_configs)

    # ===== 5. 组装中间件栈 =====
    # 自定义中间件
    from .middlewares.sandbox_health import SandboxHealthMiddleware
    from .middlewares.context_injection import ContextInjectionMiddleware
    from .middlewares.skills_sync import SkillsSyncMiddleware
    from .middlewares.user_skills_restore import UserSkillsRestoreMiddleware
    from .middlewares.tools_summarization import ToolsSummarizationMiddleware
    from .middlewares.memory_update import MemoryUpdateMiddleware
    from .middlewares.memory_consolidation import MemoryConsolidationMiddleware
    from .middlewares.warm_memory import WarmMemoryMiddleware, WARM_MEMORY_SLOT
    from .middlewares.sandbox_breaker import SandboxCircuitBreakerMiddleware
    # Harness 阶段状态机 + 评审器（真 Harness 架构核心）
    from .harness import HarnessPhaseMiddleware, load_harness_config
    from .middlewares.review_gate import ReviewExecutionGate, SafeRubricMiddleware

    # 读取 Harness DSL 配置中的评审迭代上限
    _harness_config = load_harness_config()
    _review_cfg = _harness_config.get("review", {}) if isinstance(_harness_config, dict) else {}
    _review_max_iterations = _review_cfg.get("max_iterations", 3)
    from .jev_router import build_review_router
    router_llm = build_review_router(_review_cfg)

    # 注意：create_deep_agent 内部已自动添加：
    # 装在最贴近模型请求的一层。
    # 官方中间件:
    # # ─── 1. 核心底座中间件（Core Scaffolding） ───
    # 1. FilesystemMiddleware
    #    # 提供沙箱/本地文件系统的底层读写执行工具（ls/cat/write/edit）与权限校验
    #
    # 2. SubAgentMiddleware
    #    # 提供 task 工具，支持向子 Agent 委派任务与上下文隔离
    #
    # 3. _ParentNotifierMiddleware
    #    # 子 Agent 向上级汇报状态的通信通道
    #
    # 4. SummarizationMiddleware
    #    # 会话过长时自动触发摘要，并提供 compact_conversation 工具防爆上下文
    #
    # 5. PatchToolCallsMiddleware
    #    # 修复部分大模型（如 DeepSeek/Claude）在 Tool Call 格式解析上的边缘格式 Bug
    #
    # 6. TodoListMiddleware
    #    # 注入 write_todos 工具与任务规划 Prompt 说明书
    #
    # # ─── 2. 外部自定义中间件的插入锚点 ───
    # # 【你传入的自定义中间件：HarnessPhase、ProgressiveSkills 等就是插在这个位置！】
    #
    # # ─── 3. 提示词与记忆增强中间件 ───
    # 7. SkillsMiddleware
    #    # 官方技能加载中间件（本项目设为 None，由自定义的 ProgressiveSkills 替代）
    #
    # 8. PromptCachingMiddleware
    #    # 对系统提示词做前缀缓存标记（为 Anthropic / 兼容 API 节省 Token 费用）
    #
    # 9. MemoryMiddleware
    #    # 官方静态记忆注入中间件（将 AGENTS.md 读入提示词中）
    #
    # # ─── 4. 安全、中断与兜底尾部中间件（Tail） ───
    # 10. HumanInTheLoopMiddleware
    #     # 人工审批中断中间件（拦截需要审批的敏感工具，如 order_create）
    #
    # 11. UnsupportedContentMiddleware
    #     # 过滤不支持的多媒体、特殊格式输入
    #
    # 12. ModelRefusalMiddleware
    #     # 捕获大模型拒绝回答（Refusal）的安全错误并做优雅降级
    # 我们只需添加自定义中间件 + 调用限制
    # 沙箱重建后不仅替换底层 Docker 连接，还要重新填充项目、基础 Skills 和
    # 用户持久化 Skills。两个中间件持有同一个稳定 Proxy，缓存必须一起失效。
    skills_sync_middleware = SkillsSyncMiddleware(
        skills_dir=SKILLS_DIR,
        sandbox_backend=sandbox_backend if managed_sandbox else None,
    )
    user_skills_restore_middleware = UserSkillsRestoreMiddleware(
        store=store,
        user_id=user_context.user_id,
        sandbox_backend=sandbox_backend if managed_sandbox else None,
    )

    def _rehydrate_after_sandbox_rebuild(_replacement) -> None:
        # 1. 重新打包上传项目代码
        bootstrap_sandbox(sandbox_backend)
        # 2. 撕毁“内置技能”的缓存记录
        skills_sync_middleware.invalidate()
        # 3. 撕毁“用户自定义技能”的缓存记录
        user_skills_restore_middleware.invalidate()
        # 4. 立即强制把宿主机内置技能重新灌入新容器
        skills_sync_middleware.sync_now()
        # 5. 立即强制把 MongoDB 里的用户自定义技能重新灌入新容器并 pip install
        user_skills_restore_middleware.restore_now()

    # 中间件是被编织进这个 LangGraph 图的执行节点里的钩子
    # before_model / after_model：分别在每次模型节点执行前/后运行。
    # wrap_model_call：包裹住模型调用本身，可以修改请求、重试或短路。
    # before_agent / after_agent：在整个 Agent 循环开始前和完全结束后各运行一次。
    # 有两种方式类继承和修饰器 两种方式差不多这里用的是类继承
    # 修饰器的话 你在对应方法上面写一个例如@wrap_model_call即可,方法名任意
    middlewares = [
        # 在你使用的 LangChain / Deep Agents 框架中，列表的位置本身就是语义，顺序变化会直接改变中间件的执行逻辑、包裹层级以及最终效果。
        # --- 自定义中间件（执行顺序 1→7）---
        SandboxHealthMiddleware(
            sandbox_manager=sandbox_manager_instance,
            user_id=user_context.user_id,
            sandbox_backend=sandbox_backend if managed_sandbox else None,
            on_rebuild=_rehydrate_after_sandbox_rebuild if managed_sandbox else None,
            check_interval=SANDBOX_HEALTH_CHECK_INTERVAL_SECONDS,
        ),                                                            # 1. 沙箱健康检查 + 重建
        # 阶段状态机与评审编排中间件
        HarnessPhaseMiddleware(router_model=router_llm),               # 2. 混合路由 + rubric 注入
        ContextInjectionMiddleware(user_context=user_context),        # 3. 用户上下文注入
        WarmMemoryMiddleware(store=store, user_id=user_context.user_id), # 拿偏好装到提示词
        ProgressiveSkillsMiddleware(skill_catalog),
        # 下面两个中间件是把skill同步到沙箱的代码
        skills_sync_middleware,                                       # 4. 基础 Skills 同步
        user_skills_restore_middleware,                               # 5. 用户 Skills 恢复
        ToolsSummarizationMiddleware(),                               # 6. 摘要监控
        # 有三种记忆,语义记忆(MemoryUpdateMiddleware),情节记忆(MemoryConsolidationMiddleware)
        # 以及程序记忆(remember工具),只有程序记忆(read_memory()工具)使用了keeper的分数检索search()
        MemoryUpdateMiddleware(store=store, user_id=user_context.user_id),      # 7. 偏好提取存储（WARM）
        MemoryConsolidationMiddleware(store=store, user_id=user_context.user_id),   # 7.5 情节归档 + 遗忘（COLD）
        SandboxCircuitBreakerMiddleware(failure_threshold=3, recovery_timeout=60),  # 8. 熔断器
        # --- Harness 评审器（RubricMiddleware）---
        # 收到 rubric 后，grader 子Agent 结构化产出 satisfied/needs_revision/failed，
        SafeRubricMiddleware(model=grader_llm, max_iterations=_review_max_iterations),
        ReviewExecutionGate(_harness_config),  # after_agent 逆序：先升级，再审查(洋葱模型)

        # --- 框架内置中间件（调用限制）---
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS),          # 模型调用上限
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS),            # 工具调用上限
    ]

    # ===== 6. 构建系统提示词 =====
    # preferences 是老版本的静态占位符；而真正的最新偏好确实是动态存在 warm_memory 里的。
    system_prompt = MAIN_SYSTEM_PROMPT.format(
        user_id=user_context.user_id,
        username=user_context.username,
        preferences="以本次动态加载的记忆为准",
        warm_memory=WARM_MEMORY_SLOT,
    )
    if DEFAULT_MEMORY_CONFIG.memory_tools_enabled:
        system_prompt += MEMORY_USAGE_PROMPT
    # 注入子Agent委派上下文协议
    if delegation_prompt:
        system_prompt += delegation_prompt

    # ===== 6.5 仅注入动态目录占位符，正文/资源由只读工具按需返回 =====
    system_prompt += "\n" + SLOT

    # ===== 7. 创建 Agent =====
    # 自定义渐进披露支持中文路径；不再把全部 SKILL.md 放入系统提示。
    # skills=None 防止框架与本项目双重加载；文件仍由原同步/恢复中间件准备。
    agent = create_deep_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        middleware=middlewares,
        subagents=subagents if subagents else None,
        skills=None,
        memory=[str(MEMORY_DIR / "AGENTS.md")],
        backend=composite_backend,
        interrupt_on=INTERRUPT_ON_TOOLS,# 需要人工介入的工具
        checkpointer=checkpointer,
        store=store,
        name="procurement-main-agent",
    )

    agent_logger.info(
        f"Main agent created: {len(all_tools)} tools, "
        f"{len(subagents)} subagents, {len(middlewares)} middlewares"
    )
    return agent
