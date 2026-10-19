import json
import re
from typing import Any, Dict, List, Optional

import httpx
from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion

from ..config import config

# 未显式指定 provider 时：配置了 base_url 按 OpenAI 兼容端点处理，否则按 OpenAI 官方处理
COMPAT_PROVIDER = "openai_compatible"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30

# AI 客户端缓存：按节点连接参数复用实例，避免每次请求重建连接池
_ai_clients: Dict[str, AnyLLM] = {}


# 代理
def get_proxy(open_proxy: bool = True) -> Optional[str]:
    if not open_proxy or not config.rss_proxy:
        return None
    return f"http://{config.rss_proxy}"


# 取节点使用的 provider 名称
def get_node_provider(node: Dict[str, Any]) -> str:
    provider = node.get("provider")
    if provider:
        return provider
    return COMPAT_PROVIDER if node.get("base_url") else DEFAULT_PROVIDER


# 获取（并缓存）节点对应的 any-llm 客户端
def get_ai_client(node: Dict[str, Any]) -> AnyLLM:
    provider = get_node_provider(node)
    proxy = get_proxy(node.get("use_proxy"))
    cache_key = json.dumps(
        [provider, node.get("base_url"), node.get("key"), proxy], sort_keys=True
    )

    client = _ai_clients.get(cache_key)
    if client is not None:
        return client

    client_kwargs: Dict[str, Any] = {}
    if proxy:
        # 代理绑定在客户端上，按节点隔离，多节点并发时不会互相覆盖
        client_kwargs["http_client"] = httpx.AsyncClient(proxy=proxy)

    if provider == COMPAT_PROVIDER:
        # 中转站、自部署服务等任意 OpenAI 兼容端点，参数原样透传，不做厂商适配
        if not node.get("base_url"):
            msg = "provider 为 openai_compatible 的节点必须配置 base_url"
            raise ValueError(msg)
        client = AnyLLM.create_openai_compatible(
            name="elf-rss-node",
            api_base=node["base_url"],
            api_key=node.get("key") or None,
            **client_kwargs,
        )
    else:
        client = AnyLLM.create(
            provider,
            api_key=node.get("key") or None,
            api_base=node.get("base_url") or None,
            **client_kwargs,
        )

    _ai_clients[cache_key] = client
    return client


# 构建节点请求参数
def build_ai_request(
    node: Dict[str, Any], messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": node.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "timeout": DEFAULT_TIMEOUT,
    }
    # 节点级额外参数原样写入请求体：同一模型在不同服务商要求的参数不一样，不做白名单校验
    if node.get("extra_params"):
        request["extra_body"] = node["extra_params"]
    return request


# 提取回复文本，非流式调用返回的一定是 ChatCompletion
def get_completion_text(response: Any) -> Optional[str]:
    completion: ChatCompletion = response
    return completion.choices[0].message.content


# 获取正文
def get_summary(item: Dict[str, Any]) -> str:
    summary: str = (
        item["content"][0]["value"] if item.get("content") else item["summary"]
    )
    return f"<div>{summary}</div>" if re.search("^https?://", summary) else summary


def get_author(item: Dict[str, Any]) -> str:
    return item.get("author", "")
