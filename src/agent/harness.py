"""
Harness 编排模块（真 Harness 架构核心）

职责：
1. HarnessPhaseState — 扩展 LangGraph 状态（phase / plan / review_result）
2. load_harness_config — 加载声明式 DSL 配置（harness_config.yaml）
3. build_rubric / detect_task_type — 按任务类型生成评审标准
4. HarnessPhaseMiddleware — 阶段状态机 + rubric 注入

设计说明：
- 阶段（phase）不再是 chat.py 里的正则猜测，而是 graph state 中的结构化字段，
  由中间件在钩子中写入，可被 checkpoint 持久化、可被 values 流读取。
- 评审（review）交给框架原生 RubricMiddleware：注入 rubric 后，
  其 grader 子Agent 会结构化产出 satisfied / needs_revision / failed 判定，
  needs_revision 时自动打回模型重做，形成真正的 Review 回路。
"""
# 我来说一下整个流程,包括使用skill创建todos,包装为plan,以及审批流程
# 我们知道before_model到after_model是一个循环体,这个过程中全局state会一直更新,即HarnessPhaseState会一直更新
# 我们首先搞清楚各个主要中间件的触发顺序:
# 用户输入：“帮我核对物料 P001 的各仓库存，如果总数低于 100 就生成采购建议”。
#     ↓
# SandboxHealthMiddleware.before_agent
#     ↓            └─ 测 Docker 沙箱容器是否存活（30 秒防抖）。若容器挂死则秒级热替换拉起新容器，确保底层物理环境健康可用。
# HarnessPhaseMiddleware.before_agent
#     ↓          └─ 判定是否需要审查，动态拼装质检标准，塞入 state["rubric"],置初始阶段为 state["phase"] = "planning"。
# ProgressiveSkillsMiddleware.wrap_model_call
#     ↓             └─ 把技能目录塞进系统提示词
# SkillsSyncMiddleware.before_agent
#     ↓             └─ 把宿主机项目里的内置 Skill 代码包放进沙箱
# TodoListMiddleware.wrap_model_call
#     ↓              └─ 把 write_todos 工具的说明书拼进 System Prompt。
# 大模型第 1 次思考：制定施工清单
# write_todos(todos=[{"content": "运行技能核算物料 P001 真实库存", "status": "in_progress"},
# {"content": "判断缺口并生成采购建议", "status": "pending"}])
#     ↓                   └─ 把大模型刚刚构思的两个待办事项写入 state["todos"]。(没写before_model)
# HarnessPhaseMiddleware.after_model
#     ↓                  └─ 中间件拦截到 state["todos"],包装为Plan,置阶段为 state["phase"] = "executing"。
# 大模型第 2 次思考：按需查阅 Skill 详情(从before_model开始循环)
#     ↓                  └─ 大模型在技能目录里看到 inventory_checker 符合需求,于是调用工具：read_skill(name="inventory_checker")。
# 大模型第 3 次思考：在沙箱中真正运行 Skill
#     ↓                 └─ 大模型根据查阅到的 SOP，调用沙箱命令执行工具
# 大模型第 4 次思考：更新施工清单进度
#     ↓                 └─ 模型拿到真实数据（60 件确实低于 100），再次调用 write_todos：把第 1 步状态更新为 completed，第 2 步状态更新为 in_progress。
# HarnessPhaseMiddleware.after_model
#     ↓                 └─ 拦截到最新的 todos，调用 _build_plan_from_todos,更新Plan
# 大模型第 5 次思考：输出最终分析报告并交卷
#     ↓                 └─ 大模型根据 Skill 返回的真实数据生成最终文本答卷
#                       └─ 大模型本次没有调用任何工具，大模型思考循环结束，准备退出
# SafeRubricMiddleware.after_agent
#     ↓                 └─拉起独立的 Grader 质检模型，拿出第 4 步存好的 state["rubric"]进行校验输出,检合格，写入私有状态 state["_rubric_status"] = "satisfied"。
# HarnessPhaseMiddleware.after_agent
#                         └─ 读取上一步 _rubric_status，生成最终的质检报告卡 state["review_result"]。
#                         └─最后一次同步并固化 state["plan"] 存入数据库检查点,置阶段为终态 state["phase"] = "result"。
# Grader 质检模型:专门负责挑刺判卷、和主模型完全隔离的独立法官子 Agent,注册SafeRubricMiddleware时传入
# 系统在 System Prompt 里给它注入了极其严厉的判官原则：
#   你的唯一职责：判断 <transcript>（执行记录）是否满足 <rubric>（考核标准）里的每一项。
#   有罪推定 / 保守原则（Be conservative）：只要你在记录里找不到确凿证据的，一律判定为不合格（passed=False）！
#   防被主模型欺骗：主模型说的话可能是假话，只能把它的执行记录当成“供词”，绝不能轻信！
# Grader 审查完后，被底层接口强制约束为输出以下 JSON 结构
# {
#   "result": "needs_revision",
#   "explanation": "金额格式未达标，被打回。",
#   "criteria": [
#     {"name": "真实工具证据", "passed": true},
#     {"name": "金额保留两位小数", "passed": false, "gap": "金额 500 未保留两位小数"}
#   ]
# }
# RubricResult = Literal[
#     "satisfied",              # 1. 质检通过
#     "needs_revision",         # 2. 打回重写
#     "failed",                 # 3. 任务失败
#     "max_iterations_reached", # 4. 重试超限（强行熔断）
#     "grader_error"            # 5. 质检员报错
# ]
# 如果在倒数第二步,质检模型退回要求重新跑一轮,则在HarnessPhaseMiddleware的after_agent里updates["phase"] = Phase.reviewing.value变成审查阶段
# 执行流被打回给大模型时，大模型收到了来自质检员的消息：【rubric_grader】：金额 500 未保留两位小数，被打回！
# 在官方的提示词 /todo_.py 第 132 行 中明确教了大模型"不要害怕随时修改待办清单。新的情况出现时，随时更新、重开或增加待办项。"
# 如果大模型需要重新调用工具核实数据，它会立刻再次调用 write_todos，把之前的清单全量替换，增加一个步骤：
# {"content": "重新校验并按两位小数格式化金额", "status": "in_progress"}
# 那你问了,如果需要审批在rubric中间件里直接就跳转了HarnessPhaseMiddleware怎么判断needs_revision?
# 因为rubric中间件有禁止跳转的逻辑
# class SafeRubricMiddleware(RubricMiddleware):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # 写操作评审器强制限制：最多只审 1 次
#         self._write_reviewer = RubricMiddleware(**{**kwargs, "max_iterations": 1})
#
#     @hook_config(can_jump_to=["model"])
#     def after_agent(self, state, runtime):
#         #  核心：如果大模型本轮执行了“写操作”（比如调用了创建订单、入库、删除等工具）
#         if state.get("_review_write_attempted"):
#             return self._write_reviewer.after_agent(state, runtime)  # 只审批 1 次，不准再跳
#         return super().after_agent(state, runtime)
# 如果大模型刚才已经向真实 ERP 发起了写入操作（比如扣减了库存、发起了订单），
# 如果质检说格式不合格，绝对不能允许它 jump_to="model" 自动重新跑一遍,否则会导致真实业务数据被重复扣减、订单被重复创建
# 此时HarnessPhaseMiddleware把阶段标记为 reviewing，把打回原因写进 review_result，停止自动循环，后端通过 SSE 把事件推给前端，然后直接调用 yield sse_event("done") 结束本次会话流
# 所以之后应该把项目改成变更集暂存与两阶段提交模式,即先把数据库修改方案做缓存,然后审批,然后落库

# 这个项目有大小循环
# 小循环:react,langchain底层有tools_to_model方法判断模型是否调用了工具,调用了就回到第一个before_model
# 大循环:模型自以为答完了,质检模型要求打回重新做,通过@hook_config(can_jump_to=["model"])实现
# 只有你声明了 can_jump_to，真正在编译图时把 model 连上有向边， langchain/agents/factory.py 的 _add_middleware_edge 函数（它读取了 rubric.py 上的 @hook_config 注解）才会把 'model' 加入合法的目的地列表
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware, Runtime
from langchain.agents.middleware.types import AgentState, PrivateStateAttr

from .schema import Phase, ReviewResult, Plan, PlanStep
from .log_utils import agent_logger

# DSL 配置文件路径
HARNESS_CONFIG_PATH = Path(__file__).parent / "harness_config.yaml"


class HarnessPhaseState(AgentState):
    """Harness 扩展状态：在 DeepAgentState 基础上增加阶段/计划/评审字段。

    通过 HarnessPhaseMiddleware.state_schema 声明，由框架自动合并进 graph state，
    因此这些字段会出现在 checkpoint 和 values 流中（可观测、可恢复）。
    """

    phase: NotRequired[str]
    """当前阶段：planning / executing / reviewing / result（见 schema.Phase）"""

    plan: NotRequired[dict]
    """结构化任务规划（schema.Plan 序列化后的 dict）"""

    review_result: NotRequired[dict]# 1. 最终评审结果报告（供前端展示）
    """评审结果（schema.ReviewResult 序列化后的 dict）"""

    _harness_rubric_source: NotRequired[Annotated[str, PrivateStateAttr]]
    """记录 rubric 是否由本中间件自动注入，用于跨 checkpoint 清理旧 rubric。"""

    review_decision: NotRequired[dict]  # 2. 本轮路由决策（为什么审/为什么跳过）
    _review_start: NotRequired[int]  # 3. 当前轮次消息切片起点索引
    _review_write_attempted: NotRequired[bool]  # 4. 是否尝试过增删改写操作（防重复写入）
    _review_signals: NotRequired[list[str]]  # 5. 运行时异常/工具调用特征信号


# 配置缺失时的兜底（避免文件缺失导致整个 Agent 启动失败）
_DEFAULT_CONFIG: dict[str, Any] = {
    "review": {"max_iterations": 3, "model": None},
    "task_types": {},
    "rubrics": {"default": "1. 完整回答用户问题，未遗漏关键信息\n2. 数据来自真实工具调用\n3. 输出格式规范"},
}


def load_harness_config(config_path: Path | None = None) -> dict[str, Any]:
    """加载 Harness DSL 配置。

    Args:
        config_path: 配置文件路径，默认使用模块目录下的 harness_config.yaml。

    Returns:
        配置字典。文件缺失或解析失败时回退到 _DEFAULT_CONFIG（保证降级可用）。
    """
    path = config_path or HARNESS_CONFIG_PATH
    if not path.exists():
        agent_logger.warning(f"Harness config not found at {path}, using defaults")
        return _DEFAULT_CONFIG

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config if isinstance(config, dict) else _DEFAULT_CONFIG
    except Exception as e:
        agent_logger.error(f"Failed to load harness config: {e}, using defaults")
        return _DEFAULT_CONFIG


def detect_task_type(messages: list[Any], config: dict[str, Any]) -> str:
    """根据用户消息识别任务类型，用于选择对应的评审标准。

    规则：取最近一条 HumanMessage 的文本，按 task_types 的关键词命中判定。
    多个类型命中时取关键词命中数量最多者；无命中返回 "default"。
    """
    task_types = config.get("task_types", {})
    if not task_types:
        return "default"

    # 收集用户消息文本（取最近的人类消息）
    user_text = ""
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            user_text = str(getattr(msg, "content", "") or "")
            break

    if not user_text:
        return "default"

    best_type, best_score = "default", 0
    for task_type, spec in task_types.items():
        keywords = spec.get("keywords", []) if isinstance(spec, dict) else []
        score = sum(1 for kw in keywords if kw in user_text)
        if score > best_score:
            best_type, best_score = task_type, score

    return best_type


# 这些词意味着请求通常需要工具调用、业务数据核验或多步执行，适合进入
# grader。它们是确定性的本地规则，不会为了判断“是否审查”再额外调用一次模型。
_DEFAULT_REVIEW_TRIGGER_KEYWORDS = (
    "查询", "搜索", "获取", "列出", "统计", "分析", "对比", "比较", "评估",
    "报表", "报告", "图表", "下单", "采购", "订单", "审批", "创建", "修改", "删除",
    "库存", "入库", "出库", "盘点", "预警", "供应商", "供货", "价格", "报价",
    "生成", "导出", "下载", "上传", "安装", "配置", "执行", "调用", "脚本", "代码",
    "文件", "批量", "验证", "检查", "审查", "复核", "评审", "验收", "完整", "详细", "请帮我做",
)
# 不需要进审查
_DEFAULT_TRIVIAL_PATTERNS = (
    re.compile(
        r"^(?:你好|您好|嗨|哈喽|hello|hi|hey|早上好|晚上好|晚安|谢谢|感谢|多谢|"
        r"在吗|你是谁|再见|拜拜|好的|好哒|收到|明白|了解|ok|okay|行|嗯|嗯嗯)"
        r"[!！。,.，、\s~～]*$",
        re.IGNORECASE,
    ),
)
# 不需要进审查
_DEFINITION_PATTERN = re.compile(
    r"^(?:什么是|什么叫|何为|如何理解|请解释|解释一下).*"
    r"[？?。.!！\s]*$|.*(?:是什么|什么意思|含义|定义)[？?。.!！\s]*$",
    re.IGNORECASE,
)


def latest_human_text(messages: list[Any] | None) -> str:
    """提取当前请求最近一条人类消息的纯文本内容。"""
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") != "human":
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            # 多模态消息只取文本块；图片本身不应让简单请求触发 grader。
            content = " ".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return re.sub(r"\s+", " ", str(content)).strip()
    return ""

# 正则初筛
def should_use_grader(messages: list[Any] | None, config: dict[str, Any]) -> bool:
    """判断本次请求是否需要调用 grader。

    ``review.mode`` 支持三种模式：

    * ``auto``（默认）：问候/感谢等短对话放行，业务查询和多步任务审查；
    * ``always``：兼容旧行为，每次有用户输入都审查；
    * ``never``：临时关闭审查。

    自动模式只做本地字符串判断，避免为了省 grader 调用而新增一次 LLM 调用。
    """
    review_config = config.get("review", {}) if isinstance(config, dict) else {}
    review_config = review_config if isinstance(review_config, dict) else {}
    mode = str(review_config.get("mode", "auto")).strip().lower()
    text = latest_human_text(messages)

    if mode == "never":
        return False
    if mode == "always":
        return bool(text)
    if not text:
        return False

    for pattern in _DEFAULT_TRIVIAL_PATTERNS:
        if pattern.fullmatch(text):
            return False

    # 纯概念解释不需要访问 ERP 数据，也不必为它额外启动 grader；带有
    # “查询/分析/生成”等动作的长请求仍会在下面的触发规则中进入审查。
    if re.fullmatch(r"(?:什么是[\w]{1,12}|[\w]{1,12}是什么)[？?。\s]*", text):
        return False

    configured_keywords = review_config.get("trigger_keywords")
    if isinstance(configured_keywords, list):
        trigger_keywords = tuple(str(item) for item in configured_keywords if item)
    else:
        trigger_keywords = _DEFAULT_REVIEW_TRIGGER_KEYWORDS

    # 任务类型关键词（analysis/order/inventory/supplier）也属于强触发条件。
    if detect_task_type(messages or [], config) != "default":
        return True
    if any(keyword in text for keyword in trigger_keywords):
        return True

    # 没有明确业务动作但内容较长时，通常是多约束、多步骤请求，进入审查。
    try:
        min_length = int(review_config.get("min_length", 50))
    except (TypeError, ValueError):
        min_length = 50
    return len(text) >= max(1, min_length)


def build_rubric(task_type: str, config: dict[str, Any]) -> str:
    """按任务类型生成评审标准（rubric）字符串。

    RubricMiddleware 收到该字符串后，会由 grader 子Agent 逐条对照
    执行记录（transcript）打分，驱动"是否重做"的判定。
    """
    rubrics = config.get("rubrics", {})
    rubric = rubrics.get(task_type) or rubrics.get("default", "")
    return rubric.strip()

# todos(字典列表)由TodoListMiddleware产生,需要包装为Plan
def _build_plan_from_todos(todos: list | None) -> dict | None:
    """将 TodoListMiddleware 的 todos 转换为结构化 Plan（让 Plan/PlanStep 模型真正生效）。

    todos 格式（框架产出）: [{content: str, status: "pending"|"in_progress"|"completed"}]
    Plan 格式（Harness 结构化）: {steps: [{id, content, status, result}], created_at, current_step}

    Returns:
        Plan.model_dump() 或 None（todos 为空时）
    """
    if not todos:
        return None

    steps: list[PlanStep] = []
    current_step: str | None = None
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            continue
        content = t.get("content", "")
        status = t.get("status", "pending")
        step_id = f"step-{i}"
        steps.append(PlanStep(id=step_id, content=content, status=status))
        if status == "in_progress" and current_step is None:
            # 记录当前正在执行的plan_step
            current_step = step_id

    return Plan(steps=steps, current_step=current_step).model_dump()

# 核心思想是：把 Agent 的执行过程规范为一个显式的结构化状态机（State Machine），
# 并将 phase（阶段）、plan（规划）、review_result（评审结果）作为原生字段写入 LangGraph 的 state 中。
class HarnessPhaseMiddleware(AgentMiddleware):
    """Harness 阶段状态机中间件。

    职责：
    - before_agent: 按需识别请求 → 仅对业务/多步任务注入 rubric → 置 phase=planning
    - after_agent:  置 phase=result（终态）

    阶段字段写入 state，供 chat.py 从 values 流读取并向前端发射结构化 SSE 事件，
    替代原先对 LLM 输出文本的正则猜测。
    """

    state_schema = HarnessPhaseState
    # router_model由JevRouter提供
    def __init__(self, config_path: Path | None = None, router_model=None):
        self._config = load_harness_config(config_path)
        from .review_policy import ReviewPolicy
        self._review_policy = ReviewPolicy(self._config, router_model)
        self.tools = []

    @property
    def name(self) -> str:
        return "HarnessPhaseMiddleware"

    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """执行前：按需注入 rubric(业务评审标准) + 置初始阶段为 planning。

        ``RubricMiddleware`` 本身是 state-driven 的：只有 state 中存在 rubric
        才会调用 grader。这里不再无条件注入 default rubric，因此“你好”等
        简单请求不会产生额外的审查模型调用。
        """

        return {"phase": Phase.planning.value, **self._review_policy.start(state)}

    async def abefore_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:

        return {"phase": Phase.planning.value, **await self._review_policy.astart(state)}

    def after_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """模型返回后：跟踪阶段 + 同步结构化 plan。

        - 模型产出工具调用 → 置 phase=executing（结构性写入，非正则猜测）
        - 将 TodoListMiddleware 的 todos 同步为结构化 Plan 写入 plan 字段
        """

        updates: dict[str, Any] = {}

        messages = state.get("messages", [])
        if messages and getattr(messages[-1], "tool_calls", None):
            updates["phase"] = Phase.executing.value

        # 同步 todos → plan（让 plan 字段真正生效）
        todos = state.get("todos")
        if todos:
            plan = _build_plan_from_todos(todos)
            if plan:
                updates["plan"] = plan

        return updates or None

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """执行后：根据评审状态置终态阶段 + 构建结构化评审结果。

        说明：after_agent 钩子按逆序执行，RubricMiddleware（注册在本中间件之后）
        会先完成 grading 并更新 _rubric_status，故此处可读取其私有状态。
        """
        updates: dict[str, Any] = {}

        # 从 RubricMiddleware 的私有状态构建结构化评审结果（持久化到 checkpoint）
        # 只在本次 state 仍有 rubric 时读取 status。简单请求会清空自动 rubric，
        # 不能把上一个复杂请求的终态 review_result 误带到本次响应。
        status = state.get("_rubric_status") if state.get("rubric") else None
        if status:
            evaluations = state.get("_rubric_evaluations") or []
            last = evaluations[-1] if evaluations else {}
            review_result = ReviewResult(
                verdict=status,
                explanation=last.get("explanation", ""),
                criteria=[dict(c) for c in last.get("criteria", [])],
                iteration=last.get("iteration", 0),
            )
            updates["review_result"] = review_result.model_dump()

        # 最终同步 plan（含评审后的最终步骤状态）
        # 需要持久化
        todos = state.get("todos")
        if todos:
            plan = _build_plan_from_todos(todos)
            if plan:
                updates["plan"] = plan

        # 阶段流转：needs_revision 表示评审未通过、将打回重做 → reviewing；
        # 其余（satisfied/failed/超限/异常/无 rubric）均为终态 → result
        if status == "needs_revision":
            updates["phase"] = Phase.reviewing.value
        else:
            updates["phase"] = Phase.result.value

        return updates
