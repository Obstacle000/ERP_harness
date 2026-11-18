# ERP-Agent (Harness Engineering)

基于 **DeepAgents / LangGraph** 架构构建的企业级采购与数据分析 AI Agent 系统，采用 **Harness 工程化护栏体系**，具备高可靠性、安全容器隔离与自愈能力。

B站:Harness Engineering企业级多Agent协同项目实战！Multi-Agent+SandBox+自我进化的Skill+人工介入-码士集团AI大模型
---

## 核心架构特性

### 1. Harness 中间件洋葱模型（Lifecycle Middlewares）
* **生命周期拦截**：采用经典洋葱圈模型，支持 `before_agent` / `after_agent`（全局生命周期）、`before_model` / `after_model`（轮次安检）与 `wrap_model_call`（贴身切面重试与缓存）。
* **动态审查回路**：通过 `can_jump_to=["model"]` 实现自动化质检（Rubric），遇到执行异常或不合格结果自动打回重做。
* **熔断与容错**：内置沙箱健康探测与熔断器（`SandboxCircuitBreaker`），防止死循环消耗 Token。

### 2. 安全 Docker 沙箱与预热池（Sandbox Manager）
* **秒级快速响应**：后台维护 Docker 容器预热池（Warm Pool），用户请求时 `< 100ms` 极速认领。
* **安全容器隔离**：根文件系统只读、`seccomp` 系统调用白名单、剥夺全部 root capability、严格限额 512MB 内存，彻底防止越权与逃逸。
* **无感热替换代理**：上层统一对接 `SandboxBackendProxy`，底层容器崩溃重建时原地热替换，上层 Agent 和工具调用完全无感。

### 3. 三层渐进式技能系统（Progressive Skills）
* **极度节省 Token**：系统提示词中仅包含技能轻量目录（几十字），不预载技能代码。
* **标准三步走**：大模型按需发现技能 $\rightarrow$ 调用 `read_skill` 阅读使用说明书 $\rightarrow$ 驱动沙箱安全执行。
* **双来源同步**：支持宿主机本地内置技能增量同步（`SkillsSync`）与 MongoDB Store 用户自定义技能恢复（`UserSkillsRestore`）。

### 4. 分层记忆与租户隔离（Memory & State）
* **冷热记忆分层**：动态注入即时上下文，自动提取长期偏好，支持情节归档。
* **多租户隔离**：状态、技能与容器绑定关系持久化到 MongoDB，支持多用户独立并发。

---

## 快速开始

### 依赖环境
* Python 3.11+
* Docker 环境（用于沙箱执行）
* MongoDB（用于长期状态与沙箱缓存）

### 安装与运行
```bash
# 1. 安装项目依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 配置 LLM API Key、MongoDB 连接串等

# 3. 启动 Web 服务
python -m src.api_view.web_main
```

---

## 项目目录结构
```text
src/
├── agent/                  # Agent 核心逻辑与编排
│   ├── backends/           # 沙箱物理驱动、管理器与代理 (Proxy)
│   ├── middlewares/        # 生命周期中间件栈 (健康检查、同步、熔断、路由)
│   ├── subagents/          # 子代理委派体系
│   ├── tools/              # 业务工具与 MCP 客户端
│   ├── harness.py          # 状态机定义与评审策略
│   ├── main_agent.py       # 主 Agent 组装装配点
│   └── progressive_skills.py# 渐进式技能目录与阅读器
├── api_view/               # Web API 与 SSE 流式服务层
├── repo/                   # MongoDB 与长期存储仓储层
└── skills/                 # 内置业务技能库 (procurement, web-scraper 等)
```
