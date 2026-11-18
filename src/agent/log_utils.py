"""
日志工具模块
结构化 JSON 日志，支持控制台打印 + 本地磁盘自动轮转存储
"""
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 1. 确保本地日志目录存在
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


class JSONFormatter(logging.Formatter):
    """标准的 JSON 格式格式化器"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # 必须用 json.dumps 才是合法的 JSON 字符串
        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger（同时输出到 控制台 + 本地磁盘文件）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = JSONFormatter()

        # 1. 控制台输出处理器（开发调试看）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 本地磁盘文件处理器（落盘存储，单个文件最大 10MB，最多保留 5 个备份）
        file_handler = RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,              # 轮转保留 app.log.1, app.log.2 ...
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 预定义模块 logger
agent_logger = get_logger("agent")
mcp_logger = get_logger("mcp")
sandbox_logger = get_logger("sandbox")
web_logger = get_logger("web")
middleware_logger = get_logger("middleware")