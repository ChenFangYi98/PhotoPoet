# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""GRPO 独立资源：只读 SFT 来源，校验后复制；绝不写回来源目录。"""
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from model.checkpoint_config import SCHEMA, load_checkpoint_config, sidecar_path, weight_sha256
from trainer.grpo_io import atomic_json, atomic_save

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / 'out/adaptive_sft/adaptive_20261027_131212_562620'
REWARD_REPO = 'internlm/internlm2-1_8b-reward'
REWARD_REVISION = '25f3593492ab4625ce00fce8c5e67802d6e702ca'
DEFAULT_REWARD = ROOT / 'model/internlm2-1_8b-reward'


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def write_checkpoint_config(weight, config, checksum=None):
    """GRPO-only robust writer; the paused SFT code/signature stays intact."""
    payload = dict(schema=SCHEMA, weight_file=Path(weight).name,
                   weight_sha256=checksum or weight_sha256(weight), model_config=config.to_dict())
    atomic_json(sidecar_path(weight), payload)
    return payload


def source_weight(value):
    source = project_path(value)
    if source.is_dir():
        source /= 'adaptive_768_moe_best.pth'
    if not source.is_file():
        raise FileNotFoundError(f'找不到 SFT 权重：{source}；请用 --checkpoint 指定 .pth 或 SFT 目录')
    if not sidecar_path(source).is_file():
        raise FileNotFoundError(f'缺少绑定配置：{sidecar_path(source)}，不能猜测正在训练模型的 MoE 路由语义')
    return source


def shared_read(path):
    """共享读写/删除句柄；Windows 的活动来源还须经 anchor_read 避开同名替换限制。"""
    if os.name != 'nt':
        return Path(path).open('rb')
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x08000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(handle)
        raise
    return os.fdopen(descriptor, 'rb')


@contextmanager
def anchor_read(path, directory):
    """先建立独立名字再只读打开，避免 Windows MoveFileEx 替换同名已打开目标失败。

    硬链接仅用于稳定读取，绝不当成训练输出。真正的快照是随后逐字节复制的独立文件。
    """
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if os.name != 'nt':
        with path.open('rb') as stream:
            yield stream
        return
    if path.drive.lower() != directory.drive.lower():
        directory = ROOT/'out/grpo/.read_anchors'
    if path.drive.lower() != directory.drive.lower():
        raise OSError('安全读取活动检查点需要同卷硬链接，请先将静态检查点放到项目所在卷')
    directory.mkdir(parents=True, exist_ok=True)
    alias = directory/(uuid.uuid4().hex+'.read_anchor')
    os.link(path, alias)
    try:
        with shared_read(alias) as stream:
            yield stream
    finally:
        alias.unlink(missing_ok=True)


def anchored_bytes(path, directory):
    with anchor_read(path, directory) as stream:
        return stream.read()


def protect_source(output, source):
    output, source_dir = Path(output).resolve(), Path(source).resolve().parent
    protected = [source_dir, (ROOT / 'out/adaptive_sft').resolve(), (ROOT / 'model').resolve(),
                 (ROOT / 'dataset').resolve()]
    for directory in protected:
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError(f'GRPO 输出目录必须独立于受保护目录：{directory}')


def snapshot_source(source, run_dir, retries=6):
    """原训练可原子替换导出；配置更新窗口/复制中途变化时丢弃本地副本并重试。"""
    source, run_dir = Path(source), Path(run_dir)
    target_dir = run_dir / 'source_snapshot'
    target_dir.mkdir(exist_ok=False)
    target = target_dir / source.name
    last_error = None
    for attempt in range(retries):
        try:
            before = anchored_bytes(sidecar_path(source), target_dir)
            payload = json.loads(before)
            with anchor_read(source, target_dir) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=4*1024*1024)
            checksum = weight_sha256(target)
            after = anchored_bytes(sidecar_path(source), target_dir)
            if before != after or checksum != payload['weight_sha256']:
                raise ValueError('SFT 正在更新导出，权重/配置暂不一致')
            sidecar_path(target).write_bytes(before)
            config, info = load_checkpoint_config(target)
            provenance = dict(path=str(source), sha256=checksum, config=info,
                              snapshot=str(target), source_mtime_ns=source.stat().st_mtime_ns)
            atomic_json(run_dir / 'source.json', provenance)
            return target, config, provenance
        except (OSError, ValueError, KeyError) as error:
            last_error = error
            target.unlink(missing_ok=True)
            sidecar_path(target).unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1)
    raise RuntimeError(f'无法取得一致的 SFT 快照，未修改来源：{last_error}')


def reward_files_ready(path):
    path = Path(path)
    try:
        index = json.loads((path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
        required = ['config.json', 'configuration_internlm2.py', 'modeling_internlm2.py',
                    'tokenization_internlm2.py', 'tokenizer.model', 'tokenizer_config.json']
        required += list(set(index['weight_map'].values()))
        if not all((path / name).is_file() for name in required):
            return False
        # 完整读取 safetensors 头，避免把半截下载误认为可用模型。
        from safetensors import safe_open
        for name in set(index['weight_map'].values()):
            with safe_open(str(path / name), framework='pt', device='cpu') as shard:
                if not list(shard.keys()):
                    return False
        return True
    except Exception:
        # safetensors 的 Rust 异常不继承 ValueError；缺失/损坏资源均交给下载器恢复。
        return False


def ensure_reward_model(path=DEFAULT_REWARD, download=True):
    path = project_path(path)
    if reward_files_ready(path):
        return path
    if not download:
        raise FileNotFoundError(f'评分模型不完整：{path}；运行 --prepare_only 自动下载')
    if path != DEFAULT_REWARD.resolve():
        raise FileNotFoundError(f'自定义评分模型路径不完整：{path}；自动下载只写入 {DEFAULT_REWARD}')
    from huggingface_hub import snapshot_download
    print(f'[download] {REWARD_REPO}@{REWARD_REVISION} -> {path}', flush=True)
    for attempt in range(5):
        try:
            snapshot_download(REWARD_REPO, revision=REWARD_REVISION, local_dir=str(path), max_workers=2,
                allow_patterns=['config*.json', 'configuration*.py', 'model*.py', 'tokeniz*',
                                'special_tokens_map.json', '*.safetensors', 'model.safetensors.index.json', 'README.md'])
            break
        except Exception:
            if attempt == 4:
                raise
            print(f'[download] 连接中断，保留已下载部分，重试 {attempt+1}/4', flush=True)
            time.sleep(2)
    if not reward_files_ready(path):
        raise RuntimeError('评分模型下载后仍不完整，不能开始训练')
    atomic_json(path / 'download_source.json', dict(repo=REWARD_REPO, revision=REWARD_REVISION))
    return path


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
