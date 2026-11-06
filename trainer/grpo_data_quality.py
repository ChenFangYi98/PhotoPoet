# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Read-only prompt quarantine shared by train and validation loading."""
def prompt_issue(messages):
    if not messages or messages[-1].get('role') != 'user':
        return 'missing_final_user'
    if not isinstance(messages[-1].get('content'), str) or not messages[-1]['content'].strip():
        return 'empty_final_user'
    for message in messages:
        if message.get('role') not in ('system', 'user', 'assistant'):
            return 'invalid_role'
        content = message.get('content')
        if not isinstance(content, str):
            return 'invalid_content'
        if '\ufffd' in content:
            return 'unicode_replacement_character'
        if any(ord(c) < 32 and c not in '\n\r\t' for c in content):
            return 'control_character'
    return None
