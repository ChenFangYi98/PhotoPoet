# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Project identity used by the demo and by new identity training samples."""
import re
import unicodedata


MODEL_NAME = 'MindChen'
CREATOR = 'chenfangyi'


def identity_text(name=MODEL_NAME, creator=CREATOR, english=False):
    if english:
        return f'I am {name}, trained and maintained by {creator}, the developer of this project.'
    return f'我是 {name}，由本项目的开发者{creator}训练和维护。'


_CHINESE = re.compile(
    r'你(?:是(?:谁|什么(?:模型|助手)|哪个(?:模型|助手)|哪位助手)|叫(?:什么(?:名字)?|啥(?:名字)?)|'
    r'的(?:名字|模型名称)是什么|的(?:开发者|作者)是谁|由谁(?:开发|训练|维护)|'
    r'是由谁(?:开发|训练|维护)的)|(?:谁(?:开发|训练|维护)了你)|'
    r'(?:请)?(?:介绍一下自己|做(?:个|一下)?自我介绍|自我介绍一下|告诉我你的名字)|'
    r'我应该怎么称呼你|(?:请)?(?:简短)?介绍一下你自己|'
    r'你(?:是不是|是)(?:由)?\s*(?:chatgpt|openai|阿里云|通义千问|qwen|mindchen)'
    r'\s*(?:(?:的|开发的|训练的)?(?:官方助手|助手|模型)?)(?:吗|么)?', re.IGNORECASE,
)
_ENGLISH = re.compile(
    r'who are you|what(?: is|\'s) your (?:name|model name)|what (?:model|assistant) are you|'
    r'who (?:made|developed|trained|maintains) you|who(?: is|\'s) your (?:developer|creator)|'
    r'(?:please )?introduce yourself|what should i call you|'
    r'are you (?:developed by |trained by |an? official )?'
    r'(?:chatgpt|openai|alibaba cloud|qwen|mindchen)(?: assistant)?', re.IGNORECASE,
)


def project_identity_reply(prompt):
    """Answer direct questions about the assistant's identity or developer."""
    normalized = unicodedata.normalize('NFKC', prompt).strip()
    clauses = [re.sub(r'^and\s+', '', part.strip(), flags=re.IGNORECASE)
               for part in re.split(r'[?!.。！？，,\n]+|\s+and\s+', normalized, flags=re.IGNORECASE)
               if part.strip()]
    if not clauses or not all(_CHINESE.fullmatch(part) or _ENGLISH.fullmatch(part) for part in clauses):
        return None
    return identity_text(english=all(_ENGLISH.fullmatch(part) for part in clauses))
