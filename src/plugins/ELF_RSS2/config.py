import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from nonebot import get_plugin_config
from nonebot.config import Config
from nonebot.log import logger
from pydantic import AnyHttpUrl, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pydantic v1 compatibility
    ConfigDict = None

DATA_PATH = Path.cwd() / "data"
JSON_PATH = DATA_PATH / "rss.json"


class ELFConfig(Config):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:

        class Config:
            extra = "allow"

    # 代理地址
    rss_proxy: Optional[str] = None
    rsshub: AnyHttpUrl = "https://rsshub.app"  # type: ignore
    # 备用 rsshub 地址
    rsshub_backup: List[AnyHttpUrl] = Field(default_factory=list)
    db_cache_expire: int = 30
    limit: int = 200
    max_length: int = 1024  # 正文长度限制，防止消息太长刷屏，以及消息过长发送失败的情况
    enable_boot_message: bool = True  # 是否启用启动时的提示消息推送
    debug: bool = (
        False  # 是否开启 debug 模式，开启后会打印更多的日志信息，同时检查更新时不会使用缓存,便于调试
    )

    zip_size: int = 2 * 1024
    gif_zip_size: int = 6 * 1024
    img_format: Optional[str] = None
    img_down_path: Optional[str] = None

    blockquote: bool = True
    black_word: Optional[List[str]] = None

    # 百度翻译的 appid 和 key
    baidu_id: Optional[str] = None
    baidu_key: Optional[str] = None
    deepl_translator_api_key: Optional[str] = None
    # AI API 翻译（默认节点）
    ai_api_key: Optional[str] = None
    ai_api_base_url: Optional[str] = None
    ai_api_model: Optional[str] = None
    ai_api_prompt: Optional[str] = None
    ai_api_extra_params: Optional[Dict] = None
    # 配合 deepl_translator 使用的语言检测接口，前往 https://detectlanguage.com/documentation 注册获取 api_key
    single_detection_api_key: Optional[str] = None

    qb_username: Optional[str] = None  # qbittorrent 用户名
    qb_password: Optional[str] = None  # qbittorrent 密码
    qb_web_url: Optional[str] = None  # qbittorrent 的 web 地址
    qb_down_path: Optional[str] = (
        None  # qb 的文件下载地址，这个地址必须是 go-cqhttp 能访问到的
    )
    down_status_msg_group: Optional[List[int]] = None  # 下载进度消息提示群组
    down_status_msg_date: int = 10  # 下载进度检查及提示间隔时间，单位秒

    pikpak_username: Optional[str] = None  # pikpak 用户名
    pikpak_password: Optional[str] = None  # pikpak 密码
    # pikpak 离线保存的目录, 默认是根目录，示例: ELF_RSS/Downloads ,目录不存在会自动创建, 不能/结尾
    pikpak_download_path: str = ""

    telegram_admin_ids: List[int] = Field(
        default_factory=list
    )  # Telegram 管理员 ID 列表，用于接收离线通知和管理机器人
    telegram_bot_token: Optional[str] = None  # Telegram 机器人的 token


config = get_plugin_config(ELFConfig)


def _parse_ai_api_nodes(cfg: ELFConfig) -> List[Dict[str, Any]]:
    """解析多节点 AI API 配置，支持 AI_API_KEY1/2/... 动态数量
    AI_API_KEY 作为默认节点，编号节点追加在其后，两者可同时配置
    """
    nodes: List[Dict[str, Any]] = []
    # 默认节点：AI_API_KEY
    if cfg.ai_api_key:
        default_use_proxy = os.environ.get("AI_API_USE_PROXY", "").lower() == "true"
        nodes.append({
            "key": cfg.ai_api_key,
            "base_url": cfg.ai_api_base_url,
            "model": cfg.ai_api_model,
            "prompt": cfg.ai_api_prompt,
            "extra_params": cfg.ai_api_extra_params,
            "use_proxy": default_use_proxy,
        })
    # 编号节点：AI_API_KEY1/2/... 追加在后面
    for i in range(1, 100):
        key = os.environ.get(f"AI_API_KEY{i}")
        if not key:
            break
        base_url = os.environ.get(f"AI_API_BASE_URL{i}", cfg.ai_api_base_url)
        model = os.environ.get(f"AI_API_MODEL{i}", cfg.ai_api_model)
        prompt = os.environ.get(f"AI_API_PROMPT{i}", cfg.ai_api_prompt)
        extra_params_raw = os.environ.get(f"AI_API_EXTRA_PARAMS{i}")
        extra_params = json.loads(extra_params_raw) if extra_params_raw else cfg.ai_api_extra_params
        use_proxy = os.environ.get(f"AI_API_USE_PROXY{i}", "").lower() == "true"
        nodes.append({
            "key": key,
            "base_url": base_url,
            "model": model,
            "prompt": prompt,
            "extra_params": extra_params,
            "use_proxy": use_proxy,
        })
    return nodes


config.ai_api_nodes = _parse_ai_api_nodes(config)
logger.debug(f"RSS Config loaded: {config!r}")
