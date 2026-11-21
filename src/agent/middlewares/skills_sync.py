"""
中间件 3: 本地技能同步到沙箱（完整文件夹级增量同步）

核心修复：
1. 使用相对路径（而非文件名）作为哈希 key，避免同名文件冲突
2. 保留完整子目录结构同步到沙箱 /skills/
3. 每次 Agent 执行前增量同步（hash 比对变更文件）
4. 处理远程已删除文件的清理
5. 以 skill 文件夹为单位计算整体哈希（含脚本和依赖）

Skill 标准结构：
  skills/
    procurement/
      web-scraper/
        SKILL.md
        scraper.py
        requirements.txt
      procurement-analysis/
        SKILL.md
"""
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, Runtime
from ..log_utils import middleware_logger


class SkillsSyncMiddleware(AgentMiddleware):
    """
    技能同步中间件（Harness — 完整文件夹级增量同步）

    职责：
    - 每次 Agent 执行前增量同步 src/skills/ → 沙箱 /skills/
    - 使用文件相对路径的 MD5 hash 实现精准增量同步
    - 保留完整目录结构（子目录不丢失）
    - 检测并清理沙箱中已被本地删除的文件
    - 以 skill 文件夹为单位校验完整性

    开发模式（LocalShellBackend）：
    - 本地执行无需同步，仅校验技能文件完整性
    """
    # 在本项目中，“技能文件缓存” 并不是一个简单的 Redis 或内存字典，而是针对“宿主机代码、Docker 沙箱网络 IO、MongoDB 数据库” 三方开销，
    # 设计的一套“两级 MD5 哈希指纹 + 沙箱代次（Generation）联动” 的高性能增量缓存体系。
    # 内置技能的增量缓存:
    #   中间件在内存中维护了一个 _file_hashes: dict[rel_path, str]。
    #   每次比对文件的 MD5,哈希没变，说明文件没被修改，直接跳过，绝不重复上传沙箱
    #   为了判断整个技能包有没有变，把文件夹内所有子文件的 相对路径 + 单文件MD5 拼接起来，再算一次总哈希
    # 用户自定义技能的恢复缓存（UserSkillsRestoreMiddleware）
    #   用户从 GitHub 下载的技能存在 MongoDB 里
    #   维护 _restored_hashes: dict[str, str] 记录沙箱中已有文件的哈希。
    #   只有当用户新下载了 Skill，或者修改了文件时，才往沙箱写入增量文件；已有文件命中缓存直接 0 毫秒放行。
    # 沙箱热替换时的“缓存瞬间失效（Cache Invalidation）”
    # 如果底层的 Docker 容器挂死了，热替换拉起了一个“全新的空容器”，缓存会怎么样？
    # 如果缓存不知道容器换了，还会以为“文件早已同步过”，结果新容器里空空如也，大模型一调用技能就会报 File Not Found
    # 本项目通过 generation（代次指针） 完美解决了这个问题
    def __init__(self, skills_dir: str | Path | None = None, sandbox_backend=None):
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._sandbox_backend = sandbox_backend
        # 文件级哈希缓存：relative_path → file_md5
        self._file_hashes: dict[str, str] = {}
        # 上一次同步的文件集合（用于检测删除）
        self._last_synced_files: set[str] = set()
        # 是否已完成首次同步
        self._first_sync_done = False
        # Stable Proxy 热替换后 generation 会递增，作为回调之外的兜底失效机制。
        self._last_sandbox_generation: int | None = None
        self.tools = []

    @property
    def name(self) -> str:
        return "SkillsSyncMiddleware"

    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """
        每次 Agent 执行前增量同步技能文件

        - 首次执行：完整同步
        - 后续执行：仅同步变更文件（hash 比对）
        - 检测并清理已删除文件
        """
        self.sync_now()
        return None

    async def abefore_agent(
        self, state: Any, runtime: Runtime
    ) -> dict[str, Any] | None:
        """避免在 FastAPI 的 SSE 事件循环内执行 Docker 文件 I/O。"""
        await asyncio.to_thread(self.sync_now)
        return None

    def sync_now(self) -> None:
        """立即执行一次同步，供沙箱重建回调调用。"""
        if not self._skills_dir or not self._skills_dir.exists():
            return

        if self._sandbox_backend is not None:
            # generation:当前热替换的代次
            # 热替换代理沙箱内部替换物理沙箱
            # 调用SandboxBackendProxy的replace_backend
            generation = getattr(self._sandbox_backend, "generation", None)
            if (
                # 侦测到底层物理容器被热替换了（代次变了）
                generation is not None
                and generation != self._last_sandbox_generation
            ):
                self.invalidate()# 一键清空所有已同步缓存字典
            self._sync_to_sandbox()# # 增量强制重灌，把技能重新灌进新容器
            self._last_sandbox_generation = generation
        else:
            self._validate_local_skills()

    def _validate_local_skills(self):
        """校验本地技能文件完整性"""
        skill_count = 0
        for skill_md in self._skills_dir.rglob("SKILL.md"):
            skill_count += 1
            skill_dir = skill_md.parent
            # 验证 skill 目录结构
            has_scripts = any(skill_dir.glob("*.py"))
            has_deps = (skill_dir / "requirements.txt").exists()
            middleware_logger.debug(
                f"Skill validated: {skill_dir.name} "
                f"(scripts={has_scripts}, deps={has_deps})"
            )
        middleware_logger.info(
            f"Skills validated: {skill_count} skills in {self._skills_dir}"
        )

    def _sync_to_sandbox(self):
        """
        增量同步技能文件到沙箱

        算法：
        1. 扫描本地 skills 目录，获取所有文件及其相对路径
        2. 计算每个文件的 MD5 哈希
        3. 与缓存比对，仅上传变更的文件
        4. 检测本地已删除的文件，从沙箱中清除
        5. 更新哈希缓存
        """
        if not self._sandbox_backend:
            return

        # 扫描本地文件
        current_files: dict[str, str] = {}  # rel_path → hash
        for file_path in self._skills_dir.rglob("*"):
            if file_path.is_file() and not self._should_skip(file_path):
                rel_path = str(file_path.relative_to(self._skills_dir))
                file_hash = self._compute_hash(file_path)
                current_files[rel_path] = file_hash

        current_file_set = set(current_files.keys())

        was_first_sync = not self._first_sync_done

        # === 增量上传变更文件 ===
        uploaded = 0
        for rel_path, file_hash in current_files.items():
            if self._file_hashes.get(rel_path) != file_hash:
                try:
                    file_path = self._skills_dir / rel_path
                    # 文本文件读取为 str，二进制文件跳过
                    if self._is_text_file(file_path):
                        content = file_path.read_text(encoding="utf-8")
                    else:
                        content = file_path.read_bytes()

                    remote_path = f"/skills/{rel_path}"
                    self._sandbox_backend.write_file(remote_path, content)
                    uploaded += 1
                except Exception as e:
                    middleware_logger.error(f"Failed to sync {rel_path}: {e}")

        # === 清理已删除的文件 ===
        deleted = 0
        removed_files = self._last_synced_files - current_file_set
        for rel_path in removed_files:
            try:
                remote_path = f"/skills/{rel_path}"
                self._sandbox_backend.rm(remote_path)
                deleted += 1
            except Exception as e:
                middleware_logger.warning(f"Failed to remove {rel_path}: {e}")

        # === 更新缓存 ===
        self._file_hashes.update(current_files)
        # 移除已删除文件的缓存
        for rel_path in removed_files:
            self._file_hashes.pop(rel_path, None)
        self._last_synced_files = current_file_set
        self._first_sync_done = True

        # 日志
        if uploaded > 0 or deleted > 0:
            middleware_logger.info(
                f"Skills synced: {uploaded} uploaded, {deleted} deleted "
                f"(total: {len(current_files)} files)"
            )
        elif was_first_sync:
            middleware_logger.info(
                f"Skills sync: {len(current_files)} files (no changes)"
            )

    def _should_skip(self, file_path: Path) -> bool:
        """判断是否跳过文件"""
        skip_patterns = {
            "__pycache__", ".pyc", ".pyo", ".git", ".DS_Store",
            "node_modules", ".egg-info",
        }
        name = file_path.name
        return any(pat in name for pat in skip_patterns)

    @staticmethod
    def _is_text_file(file_path: Path) -> bool:
        """判断是否为文本文件"""
        text_extensions = {
            ".md", ".txt", ".py", ".yaml", ".yml", ".json",
            ".csv", ".toml", ".cfg", ".ini", ".sh", ".bash",
            ".js", ".ts", ".html", ".css", ".xml", ".rst",
        }
        return file_path.suffix.lower() in text_extensions

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """计算文件 MD5 hash"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def compute_skill_folder_hash(self, skill_dir: Path) -> str:
        """
        计算整个 skill 文件夹的哈希值（用于判断 skill 是否有变更）

        将文件夹内所有文件的哈希拼接后再哈希，作为整个 skill 的指纹。
        """
        if not skill_dir.is_dir():
            return ""
        folder_hash = hashlib.md5()
        for file_path in sorted(skill_dir.rglob("*")):
            if file_path.is_file() and not self._should_skip(file_path):
                rel = str(file_path.relative_to(skill_dir))
                file_hash = self._compute_hash(file_path)
                folder_hash.update(f"{rel}:{file_hash}".encode())
        return folder_hash.hexdigest()

    def invalidate(self):
        """手动失效缓存，强制下次完整同步（新 Skill 安装后调用）"""
        self._file_hashes.clear()
        self._last_synced_files.clear()
        self._first_sync_done = False
        self._last_sandbox_generation = None
        middleware_logger.info("SkillsSync cache invalidated, will full sync next time")
