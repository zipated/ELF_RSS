import hashlib
import random
import re
from typing import Dict, Optional

import aiohttp
import emoji
from deep_translator import DeeplTranslator, GoogleTranslator, single_detection
from nonebot.log import logger

from ..config import config


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


async def volcano_translator(
    text: str,
    api_key: str,
    model_id: str,  # 用户自定义Model ID
) -> str:
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的多语言翻译专家，精通语言学、文化差异和技术术语。您的目标是提供准确、自然且符合语境的跨语言翻译。翻译目标语言为简体中文。不需要翻译的说明和评论。文本开头的`RT `、`Re `独立保留且不作翻译，只有`RT`、`Re`、`RT `、`Re `存在时不需要翻译。未提供内容则不需要翻译。",
            },
            {
                "role": "user",
                "content": "Translate into zh:\n" + text,
            },
        ],
    }
    async with aiohttp.ClientSession() as session:
        resp = await session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(10))
        result = await resp.json()
        try:
            content = result["choices"][0]["message"]["content"]
            return "\nAI翻译(" + model_id + ")：\n" + content
        except (KeyError, IndexError) as e:
            error_msg = "\nAI引擎(" + model_id + ")翻译失败：" + str(e) + "\n"
            logger.warning(error_msg)
            raise Exception(error_msg) from e

# 翻译
async def handle_translation(content: str) -> str:
    proxies = (
        {
            "https": config.rss_proxy,
            "http": config.rss_proxy,
        }
        if config.rss_proxy
        else None
    )

    text = emoji.demojize(content)
    text = re.sub(r":[A-Za-z_]*:", " ", text)
    try:
        # 优先级 DeeplTranslator > 火山 > 百度翻译 > GoogleTranslator
        # 异常时使用 GoogleTranslator 重试
        google_translator_flag = False
        try:
            if config.deepl_translator_api_key:
                text = await deepl_translator(text=text, proxies=proxies)
            elif config.volcano_api_key and config.volcano_model_id:
                text = await volcano_translator(text=text, api_key=config.volcano_api_key, model_id=config.volcano_model_id)
            elif config.baidu_id and config.baidu_key:
                text = await baidu_translator(
                    content, config.baidu_id, config.baidu_key
                )
            else:
                google_translator_flag = True
        except Exception:
            google_translator_flag = True
        if google_translator_flag:
            text = await google_translation(text=text, proxies=proxies)
    except Exception as e:
        logger.error(e)
        text = str(e)

    text = text.replace("\\", "")
    return text
