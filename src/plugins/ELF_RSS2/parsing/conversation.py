from typing import Dict, List


# 按 RSS 源 URL 缓存 AI 翻译会话历史，同来源共享上下文以保持术语一致
MAX_HISTORY = 10  # 每个源最多保留10条历史

_conversation_cache: Dict[str, List[Dict[str, str]]] = {}


def get_conversation_history(url: str) -> List[Dict[str, str]]:
    """获取指定来源的会话历史"""
    return list(_conversation_cache.get(url, []))


def add_conversation(url: str, original: str, translated: str) -> None:
    """添加翻译记录到会话历史"""
    if url not in _conversation_cache:
        _conversation_cache[url] = []
    history = _conversation_cache[url]
    history.append({"role": "user", "content": original})
    history.append({"role": "assistant", "content": translated})
    # 限制历史长度，避免 token 溢出
    if len(history) > MAX_HISTORY * 2:
        _conversation_cache[url] = history[-(MAX_HISTORY * 2):]


def clear_conversation(url: str) -> None:
    """清除指定来源的会话历史"""
    _conversation_cache.pop(url, None)
