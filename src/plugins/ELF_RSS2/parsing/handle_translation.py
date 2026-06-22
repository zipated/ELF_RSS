import hashlib
import os
import random
import re
from typing import Any, Dict, List, Optional

import aiohttp
import emoji
import litellm
from deep_translator import DeeplTranslator, GoogleTranslator, single_detection
from nonebot.log import logger

from ..config import config
from .conversation import add_conversation, get_conversation_history
from .knowledge_base import get_knowledge_base_prompt, load_knowledge_base

# 关闭 litellm 内置的 debug 日志
litellm.suppress_debug_info = True

LANGUAGE_MAP = {
    "ar": "阿拉伯语",
    "de": "德语",
    "en": "英语",
    "eo": "世界语",
    "es": "西班牙语",
    "fi": "芬兰语",
    "fr": "法语",
    "ia": "国际语",
    "it": "意大利语",
    "ja": "日语",
    "ko": "韩语",
    "pt": "葡萄牙语",
    "zh": "中文",
}


async def baidu_translator(content: str, appid: str, secret_key: str) -> str:
    url = "https://api.fanyi.baidu.com/api/trans/vip/translate"
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5((appid + content + salt + secret_key).encode()).hexdigest()
    params = {
        "q": content,
        "from": "auto",
        "to": "zh",
        "appid": appid,
        "salt": salt,
        "sign": sign,
    }
    async with aiohttp.ClientSession() as session:
        resp = await session.get(url, params=params, timeout=aiohttp.ClientTimeout(10))
        data = await resp.json()
        try:
            content = "".join(i["dst"] + "\n" for i in data["trans_result"])
            return "\n百度翻译：\n" + content[:-1]
        except Exception as e:
            error_msg = f"百度翻译失败：{data['error_msg']}"
            logger.warning(error_msg)
            raise Exception(error_msg) from e


async def google_translation(text: str, proxies: Optional[Dict[str, str]]) -> str:
    # text 是处理过emoji的
    try:
        translator = GoogleTranslator(source="auto", target="zh-CN", proxies=proxies)
        return "\n谷歌翻译：\n" + str(translator.translate(re.escape(text)))
    except Exception as e:
        error_msg = "\nGoogle翻译失败：" + str(e) + "\n"
        logger.warning(error_msg)
        raise Exception(error_msg) from e


async def deepl_translator(text: str, proxies: Optional[Dict[str, str]]) -> str:
    try:
        lang = None
        if config.single_detection_api_key:
            lang = single_detection(text, api_key=config.single_detection_api_key)
        translator = DeeplTranslator(
            api_key=config.deepl_translator_api_key,
            source=lang,
            target="zh",
            use_free_api=True,
            proxies=proxies,
        )
        return "\nDeepl翻译：\n" + str(translator.translate(re.escape(text)))
    except Exception as e:
        error_msg = "\nDeeplTranslator翻译失败：" + str(e) + "\n"
        logger.warning(error_msg)
        raise Exception(error_msg) from e


async def _detect_lang(text: str) -> Optional[str]:
    # 检测语言并返回中文名称
    if config.single_detection_api_key:
        lang = single_detection(text, api_key=config.single_detection_api_key)
        return LANGUAGE_MAP.get(lang, lang)
    return None


def _build_system_prompt(
    node: Dict[str, Any], lang: Optional[str], kb_data: Optional[Dict[str, Any]]
) -> str:
    # 构建系统 prompt，包含自定义 prompt、语言信息和资料库
    base_prompt = node.get("prompt")
    if base_prompt:
        if lang and "{lang}" in base_prompt:
            prompt = base_prompt.format(lang=lang)
        else:
            prompt = base_prompt
    elif lang:
        prompt = f"你是一个专业的多语言翻译器，请将内容从{lang}翻译为准确、自然且符合语境的简体中文。仅输出翻译结果，不要添加任何解释、说明、补充或分析。"
    else:
        prompt = "你是一个专业的多语言翻译器，请翻译为准确、自然且符合语境的简体中文。仅输出翻译结果，不要添加任何解释、说明、补充或分析。"

    if kb_data:
        kb_prompt = get_knowledge_base_prompt(kb_data)
        if kb_prompt:
            prompt += "\n\n" + kb_prompt

    return prompt


def _build_litellm_kwargs(
    node: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    # 将节点配置映射为 litellm.acompletion() 参数
    model = node.get("model") or "gpt-4o-mini"
    base_url = node.get("base_url")
    # 自定义 endpoint 走 OpenAI 兼容协议，需要 openai/ 前缀
    if base_url and "/" not in model:
        model = f"openai/{model}"

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": node["key"],
        "timeout": 30,
    }
    if base_url:
        kwargs["base_url"] = base_url
    # litellm 参数放行白名单，强制透传非标准参数（如 thinking）
    if node.get("allowed_openai_params"):
        kwargs["allowed_openai_params"] = node["allowed_openai_params"]
    # extra_params 中的合法 litellm 参数透传
    if node.get("extra_params"):
        kwargs.update(node["extra_params"])
    return kwargs


async def _ai_translate_with_node(
    text: str,
    proxy: Optional[str],
    node: Dict[str, Any],
    rss_url: Optional[str] = None,
    kb_data: Optional[Dict[str, Any]] = None,
) -> str:
    # 使用单个 API 节点进行翻译（通过 LiteLLM）
    lang = await _detect_lang(text)
    system_content = _build_system_prompt(node, lang, kb_data)

    messages: List[Dict[str, str]] = []

    # 注入同来源的会话历史，保持术语一致性
    if rss_url:
        history = get_conversation_history(rss_url)
        messages.extend(history)

    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": text})

    kwargs = _build_litellm_kwargs(node, messages)

    # 按节点配置决定是否用代理：litellm 通过 HTTP_PROXY/HTTPS_PROXY 环境变量读取代理
    old_http_proxy = None
    old_https_proxy = None
    if node.get("use_proxy") and proxy:
        old_http_proxy = os.environ.get("HTTP_PROXY")
        old_https_proxy = os.environ.get("HTTPS_PROXY")
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        model_name = node.get("model") or "unknown"
        return f"\nAI翻译({model_name})：\n{content}"
    except Exception as e:
        model_name = node.get("model") or "unknown"
        error_msg = f"\nAI({model_name})翻译失败：{e}\n"
        logger.warning(error_msg)
        raise Exception(error_msg) from e
    finally:
        # 恢复代理环境变量
        if old_http_proxy is not None:
            os.environ["HTTP_PROXY"] = old_http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
        if old_https_proxy is not None:
            os.environ["HTTPS_PROXY"] = old_https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)


async def ai_translator(
    text: str,
    proxy: Optional[str],
    rss_name: Optional[str] = None,
    rss_url: Optional[str] = None,
) -> str:
    # 多节点 AI 翻译，按顺序尝试直到成功
    kb_data = None
    if rss_name:
        kb_data = load_knowledge_base(rss_name)

    last_error: Optional[Exception] = None
    for node in config.ai_api_nodes:
        try:
            result = await _ai_translate_with_node(
                text=text,
                proxy=proxy,
                node=node,
                rss_url=rss_url,
                kb_data=kb_data,
            )
            # 翻译成功后记录会话历史
            if rss_url:
                translated = result.split("：\n", 1)[-1] if "：\n" in result else result
                add_conversation(rss_url, text, translated)
            return result
        except Exception as e:
            node_desc = node.get("base_url") or node.get("model", "default")
            logger.warning(f"AI翻译节点 [{node_desc}] 失败: {e}，尝试下一个节点")
            last_error = e
            continue

    if last_error:
        raise last_error
    raise Exception("所有AI翻译节点均失败")


# 翻译
async def handle_translation(
    content: str,
    rss_name: Optional[str] = None,
    rss_url: Optional[str] = None,
) -> str:
    proxy = config.rss_proxy
    proxies = (
        {"https": proxy, "http": proxy}
        if proxy
        else None
    )

    text = emoji.demojize(content)
    text = re.sub(r":[A-Za-z_]*:", " ", text)
    try:
        # 级联回退：AI(多节点) → DeepL → Baidu → Google
        last_error: Optional[Exception] = None

        # 1. AI 翻译（多节点，内部已按顺序尝试）
        if config.ai_api_nodes:
            try:
                text = await ai_translator(
                    text=text,
                    proxy=proxy,
                    rss_name=rss_name,
                    rss_url=rss_url,
                )
                return _clean_translation(text)
            except Exception as e:
                last_error = e
                logger.warning(f"AI翻译全部失败，回退到下一级: {e}")

        # 2. DeepL
        if config.deepl_translator_api_key:
            try:
                text = await deepl_translator(text=text, proxies=proxies)
                return _clean_translation(text)
            except Exception as e:
                last_error = e
                logger.warning(f"DeepL翻译失败，回退到下一级: {e}")

        # 3. 百度翻译
        if config.baidu_id and config.baidu_key:
            try:
                text = await baidu_translator(
                    text, config.baidu_id, config.baidu_key
                )
                return _clean_translation(text)
            except Exception as e:
                last_error = e
                logger.warning(f"百度翻译失败，回退到下一级: {e}")

        # 4. Google 翻译（最后兜底）
        try:
            text = await google_translation(text=text, proxies=proxies)
        except Exception as e:
            logger.error(f"所有翻译方式均失败，最后错误: {e}")
            if last_error:
                text = str(last_error)
            else:
                text = str(e)
    except Exception as e:
        logger.error(e)
        text = str(e)

    return _clean_translation(text)


def _clean_translation(text: str) -> str:
    return text.replace("\\", "")
