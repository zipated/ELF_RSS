import json
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import litellm
from nonebot.log import logger
from pyquery import PyQuery as Pq

from ..config import DATA_PATH, config
from ..parsing.utils import get_proxy

litellm.suppress_debug_info = True

KB_PATH = DATA_PATH / "kb"

# 爬取时尝试的子页面路径
_SUB_PATHS = ["/about", "/info", "/about/", "/info/", "/about.html", "/info.html"]

_KB_GENERATE_PROMPT = """\
你是一个专业的资料库生成助手。请根据以下网页内容，为该 RSS 订阅来源生成结构化的翻译辅助资料库。

请生成 JSON 格式的资料库，包含：
1. background: 该来源的简要背景描述（2-5句话）
2. terminology: 常见专业术语的翻译对照表（至少5条，如果没有明显的专业术语则为空列表）

要求：
- 术语应包含来源语言原文和对应的中文翻译
- 如果来源已经是中文，背景描述用中文，术语列表留空
- 只输出 JSON，不要其他内容

JSON 格式示例：
{
  "background": "...",
  "terminology": [
    {"source": "English Term", "translation": "中文翻译"},
    {"source": "Another Term", "translation": "另一个翻译"}
  ]
}

网页内容：
"""


def _get_kb_path(rss_name: str) -> Path:
    return KB_PATH / f"{rss_name}.json"


def load_knowledge_base(rss_name: str) -> Optional[Dict[str, Any]]:
    # 加载指定订阅的资料库
    path = _get_kb_path(rss_name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载资料库失败 [{rss_name}]: {e}")
        return None


def get_knowledge_base_prompt(kb_data: Dict[str, Any]) -> str:
    # 将资料库数据转为注入翻译 prompt 的文本
    parts = []

    if background := kb_data.get("background"):
        parts.append(f"## 来源背景\n{background}")

    if terminology := kb_data.get("terminology"):
        terms = "\n".join(
            f"- {t['source']} → {t['translation']}" for t in terminology if t.get("source")
        )
        if terms:
            parts.append(f"## 术语对照表\n{terms}")

    if not parts:
        return ""
    return "以下是该 RSS 来源的相关背景资料，请在翻译时参考以提高准确性：\n\n" + "\n\n".join(parts)


async def _fetch_page_text(url: str, proxy: Optional[str]) -> str:
    # 抓取页面并提取文本内容
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            resp = await session.get(url, proxy=proxy)
            html = await resp.text()
            doc = Pq(html)
            doc("script").remove()
            doc("style").remove()
            text = doc.text()
            return text[:3000]
    except Exception as e:
        logger.debug(f"抓取页面失败 [{url}]: {e}")
        return ""


async def _generate_kb_with_ai(content: str, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 使用 AI（LiteLLM）从网页内容生成资料库
    prompt = _KB_GENERATE_PROMPT + content

    kwargs: Dict[str, Any] = {
        "model": node.get("model") or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "你只输出 JSON，不输出其他内容。"},
            {"role": "user", "content": prompt},
        ],
        "api_key": node["key"],
        "timeout": 30,
    }
    if node.get("base_url"):
        kwargs["base_url"] = node["base_url"]
    if node.get("extra_params"):
        kwargs.update(node["extra_params"])

    try:
        response = await litellm.acompletion(**kwargs)
        content_text = response.choices[0].message.content
        content_text = content_text.strip()
        if content_text.startswith("```"):
            content_text = content_text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content_text)
    except Exception as e:
        logger.warning(f"AI 生成资料库失败: {e}")
        return None


async def generate_knowledge_base(rss_url: str, rss_name: str) -> Optional[Dict[str, Any]]:
    # 爬取来源网站 + AI 生成资料库
    proxy = get_proxy()

    from yarl import URL

    parsed = URL(rss_url)
    base = f"{parsed.scheme}://{parsed.host}"

    logger.info(f"正在为 [{rss_name}] 生成资料库，爬取来源网站...")
    main_text = await _fetch_page_text(base, proxy)

    sub_texts = []
    for path in _SUB_PATHS:
        sub_url = base + path
        text = await _fetch_page_text(sub_url, proxy)
        if text:
            sub_texts.append(text)

    all_text = main_text
    if sub_texts:
        all_text += "\n\n补充页面内容：\n" + "\n\n".join(sub_texts)

    if not all_text.strip():
        logger.warning(f"[{rss_name}] 未能获取到有效的网页内容，跳过资料库生成")
        return None

    if not config.openapi_nodes:
        logger.warning("未配置 AI API 节点，无法生成资料库")
        return None

    kb_data = await _generate_kb_with_ai(all_text, config.openapi_nodes[0])
    if not kb_data:
        return None

    if "background" not in kb_data:
        kb_data["background"] = ""
    if "terminology" not in kb_data:
        kb_data["terminology"] = []

    KB_PATH.mkdir(parents=True, exist_ok=True)
    path = _get_kb_path(rss_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[{rss_name}] 资料库生成成功")
    return kb_data
