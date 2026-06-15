import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser
import litellm
from nonebot.log import logger

from ..config import DATA_PATH, config

litellm.suppress_debug_info = True

KB_PATH = DATA_PATH / "kb"

_SEARCH_API_URL = "https://open.feedcoopapi.com/search_api/web_search"

_KB_GENERATE_PROMPT = """\
你是一个专业的资料库生成助手。请根据以下 RSS 来源的信息和搜索结果，生成结构化的翻译辅助资料库。

来源信息：
- 来源标题：{feed_title}
- 来源链接：{feed_link}
- 来源描述：{feed_description}

联网搜索结果：
{search_results}

请生成 JSON 格式的资料库，包含：
1. background: 该来源的简要背景描述（2-5句话，根据标题、链接、搜索结果和你的知识描述这是什么类型的来源、主要内容方向）
2. terminology: 常见专业术语的翻译对照表（至少5条，如果没有明显的专业术语则为空列表）

要求：
- 术语应包含来源语言原文和对应的中文翻译
- 翻译保持简洁本意，不要加括号注释、解释或补充说明。例如「歌ってみた」翻译为「翻唱」而非「翻唱（歌曲翻唱视频）」，专有名词如「FANBOX」直接使用原名
- 如果来源已经是中文，背景描述用中文，术语列表留空
- 只输出 JSON，不要其他内容

JSON 格式示例：
{{
  "background": "...",
  "terminology": [
    {{"source": "English Term", "translation": "中文翻译"}},
    {{"source": "Another Term", "translation": "另一个翻译"}}
  ]
}}
"""


def _get_kb_path(rss_name: str) -> Path:
    return KB_PATH / f"{rss_name}.json"


def load_knowledge_base(rss_name: str) -> Optional[Dict[str, Any]]:
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


async def _fetch_feed_metadata(rss_url: str) -> tuple:
    # 抓取 RSS feed 提取标题/链接/描述
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
            resp = await session.get(rss_url)
            d = feedparser.parse(await resp.text())
            feed = d.get("feed", {})
            return (
                feed.get("title"),
                feed.get("link"),
                feed.get("description") or feed.get("subtitle"),
            )
    except Exception as e:
        logger.debug(f"抓取 feed 元数据失败 [{rss_url}]: {e}")
        return None, None, None


async def _search_web(query: str) -> List[Dict[str, str]]:
    # 调用火山引擎联网搜索 API
    payload = {
        "Query": query[:100],
        "SearchType": "web",
        "Count": 20,
        "NeedSummary": True,
    }
    if config.kb_search_sites:
        payload.setdefault("Filter", {})["Sites"] = config.kb_search_sites
    if config.kb_search_block_hosts:
        payload.setdefault("Filter", {})["BlockHosts"] = config.kb_search_block_hosts

    headers = {
        "Authorization": f"Bearer {config.kb_search_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(15)) as session:
            async with session.post(_SEARCH_API_URL, json=payload, headers=headers) as resp:
                data = await resp.json()
                results = data.get("Result", {}).get("WebResults", [])
                return [
                    {
                        "title": r.get("Title", ""),
                        "snippet": r.get("Summary") or r.get("Snippet", ""),
                        "url": r.get("Url", ""),
                    }
                    for r in results
                ]
    except Exception as e:
        logger.warning(f"联网搜索失败 [{query}]: {e}")
        return []


def _format_search_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "无搜索结果"
    lines = []
    for i, r in enumerate(results[:10], 1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet'][:300]}\n   {r['url']}")
    return "\n\n".join(lines)


async def _generate_kb_with_ai(
    node: Dict[str, Any],
    feed_title: str,
    feed_link: str,
    feed_description: str,
    search_results: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    prompt = _KB_GENERATE_PROMPT.format(
        feed_title=feed_title or "未知",
        feed_link=feed_link or "未知",
        feed_description=feed_description or "无",
        search_results=_format_search_results(search_results),
    )

    model = node.get("model") or "gpt-4o-mini"
    base_url = node.get("base_url")
    if base_url and "/" not in model:
        model = f"openai/{model}"

    kwargs: Dict[str, Any] = {
        "model": model,
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
        first_brace = content_text.find("{")
        last_brace = content_text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            content_text = content_text[first_brace : last_brace + 1]
        else:
            content_text = "{" + content_text + "}"
        return json.loads(content_text)
    except Exception as e:
        logger.warning(f"AI 生成资料库失败: {e}")
        return None


async def generate_knowledge_base(
    rss_url: str,
    rss_name: str,
    feed_title: Optional[str] = None,
    feed_link: Optional[str] = None,
    feed_description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = _get_kb_path(rss_name)
    if path.exists():
        logger.info(f"[{rss_name}] 资料库文件已存在，跳过生成")
        return load_knowledge_base(rss_name)

    if not config.kb_search_api_key:
        logger.warning("未配置 KB_SEARCH_API_KEY，无法生成资料库")
        return None

    if not config.ai_api_nodes:
        logger.warning("未配置 AI API 节点，无法生成资料库")
        return None

    kb_nodes = [n for n in config.ai_api_nodes if n.get("kb_enabled", True)]
    if not kb_nodes:
        logger.warning("所有 AI API 节点均已关闭资料库生成功能，无法生成资料库")
        return None

    # 1. 抓取 RSS feed 获取标题等元数据
    if not feed_title:
        title, link, desc = await _fetch_feed_metadata(rss_url)
        feed_title = feed_title or title
        feed_link = feed_link or link
        feed_description = feed_description or desc

    # 2. 用标题搜索
    search_query = feed_title or rss_name
    logger.info(f"正在为 [{rss_name}] 生成资料库，搜索: {search_query[:50]}...")
    search_results = await _search_web(search_query) if search_query else []

    # 3. AI 生成
    logger.info(f"正在为 [{rss_name}] 生成资料库（基于 AI 知识 + {len(search_results)} 条搜索结果）...")
    kb_data = await _generate_kb_with_ai(
        node=kb_nodes[0],
        feed_title=feed_title or "未知",
        feed_link=feed_link or "未知",
        feed_description=feed_description or "无",
        search_results=search_results,
    )
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


async def generate_kb_for_rss(
    rss_url: str,
    rss_name: str,
    feed_title: Optional[str] = None,
    feed_link: Optional[str] = None,
    feed_description: Optional[str] = None,
) -> bool:
    # 生成资料库（含失败处理），返回 True=成功/已存在，False=失败
    try:
        result = await generate_knowledge_base(
            rss_url=rss_url,
            rss_name=rss_name,
            feed_title=feed_title,
            feed_link=feed_link,
            feed_description=feed_description,
        )
        return result is not None
    except Exception as e:
        logger.warning(f"[{rss_name}] 资料库生成异常: {e}")
        return False


def check_kb_missing(rss_name: str) -> bool:
    # 检查资料库文件是否缺失，True=缺失
    path = _get_kb_path(rss_name)
    return not path.exists()