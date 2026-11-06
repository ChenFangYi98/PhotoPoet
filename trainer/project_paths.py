# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""项目整体搬盘后的SFT断点路径重定位；不改历史文件与模型张量。"""
from pathlib import Path, PureWindowsPath
from model.checkpoint_compat import LEGACY_PROJECT_DIR_NAMES


def relocate_project_path(value, current_root, previous_root=None):
    """只转换项目内部路径；项目外Python环境和外部数据路径保持原样。"""
    root = Path(current_root).resolve()
    old_path = PureWindowsPath(str(value))
    if not old_path.is_absolute():
        return str(value)
    if previous_root is None:
        names = {root.name.casefold(), *LEGACY_PROJECT_DIR_NAMES}
        previous_root = next((p for p in old_path.parents if p.name.casefold() in names), None)
    if previous_root is None:
        return str(value)
    try:
        relative = old_path.relative_to(PureWindowsPath(str(previous_root)))
    except ValueError:
        return str(value)
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'重定位路径超出项目目录: {value}')
    return str(candidate)


def relocate_sft_checkpoint(checkpoint, current_root):
    """只重定位内存中的路径；大小、时间戳、SHA256仍由训练入口严格校验。"""
    previous_root = checkpoint.get('project_root')
    changes = {}
    for key in ('init_weight_path', 'train_data_path', 'valid_data_path', 'save_dir'):
        old = checkpoint['args'].get(key)
        if old is not None:
            new = relocate_project_path(old, current_root, previous_root)
            checkpoint['args'][key] = new
            if str(old) != new:
                changes[key] = {'old': str(old), 'new': new}
    for info in [checkpoint.get('source_weight', {}), *checkpoint.get('data_files', {}).values()]:
        if 'path' in info:
            info['path'] = relocate_project_path(info['path'], current_root, previous_root)
    return changes
