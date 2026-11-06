# Modified by chenfangyi for the unified MindChen release (2026-11-06).
# -*- coding: utf-8 -*-
"""Multimodal Dataset Studio — 单文件版 (PySide6 + PyArrow)

依赖安装 (Windows / Python 3.10+):
    python -m pip install PySide6 pyarrow pillow

运行:
    python 多模态数据集编辑器.py

使用:
    1. 点击【新建项目】，选择编辑记录的 .sftproj 保存位置。
    2. 点击【导入 Parquet】打开现有 MindChen-V sft_i2t.parquet，或导入图片文件夹/CSV。
    3. 选择左侧样本，右侧修改多轮 conversations，Ctrl+S 保存。
    4. 点击【导出 Parquet】，生成 conversations + image_bytes 的单图数据集。

注意:
    - 源 Parquet 和原始图片不会被修改；项目只保存引用和编辑差异。
    - 源数据/图片移动或删除后，需要恢复其原有路径才能继续读取。
    - 当前版本处理单图 image_bytes (bytes 或 [bytes])，不是多图序列。
    - 使用子进程命令 python -m pip install ... 安装依赖，不需任何额外源码文件。
"""
from __future__ import annotations

import bisect
import csv
import io
import itertools
import json
import os
import sqlite3
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps
    from PySide6.QtCore import (
        Qt, QSize, QModelIndex, QThread, Signal, QAbstractTableModel
    )
    from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
        QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
        QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPlainTextEdit, QProgressDialog, QPushButton, QSpinBox,
        QSplitter, QStatusBar, QTableView, QTableWidget, QToolBar,
        QVBoxLayout, QWidget, QAbstractItemView,
    )
except ImportError as error:
    raise SystemExit(
        f"缺少运行依赖：{error}\n请在终端执行：\n"
        "python -m pip install PySide6 pyarrow pillow\n"
        "然后重新运行本文件。"
    ) from error


# ====== 项目及 Parquet 存储模块 ======
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def dumps_conv(turns: list[dict]) -> str:
    return json.dumps(turns, ensure_ascii=False, separators=(",", ":"))


def parses_conv(value: Any) -> list[dict]:
    turns = json.loads(value) if isinstance(value, str) else value
    if not isinstance(turns, list):
        raise ValueError("conversations 必须是 JSON 数组")
    if not all(isinstance(x, dict) for x in turns):
        raise ValueError("conversations 数组元素必须是对象")
    return turns


def validate_turns(turns: list[dict], image_present: bool = True) -> list[str]:
    errors = []
    if not turns:
        return ["对话为空"]
    valid_roles = {"system", "user", "assistant"}
    for i, turn in enumerate(turns, 1):
        role, content = turn.get("role"), turn.get("content")
        if role not in valid_roles:
            errors.append(f"第 {i} 轮角色不合法：{role}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"第 {i} 轮内容为空")
    if not any(t.get("role") == "user" for t in turns):
        errors.append("至少需要一条 user 消息")
    if not any(t.get("role") == "assistant" for t in turns):
        errors.append("至少需要一条 assistant 消息")
    # Some real SFT rows are text-only and use a black placeholder; not enforcing <image>.
    return errors


def image_as_bytes(value: Any) -> bytes:
    """MindChen-V loader accepts binary or [binary]; UI is single-image only."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"此编辑器只支持单图，收到 {len(value)} 张")
        value = value[0]
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes):
        raise ValueError("image_bytes 不是有效的图片二进制内容")
    return value


def placeholder_image() -> bytes:
    f = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(f, format="JPEG")
    return f.getvalue()


BLACK_IMAGE = placeholder_image()


def encode_image(path: str, size: int = 256, quality: int = 95) -> bytes:
    with Image.open(path) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
        if size > 0:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, "JPEG", quality=quality, optimize=True)
        return output.getvalue()


def make_turns(question: str, answer: str, has_image: bool = True) -> list[dict]:
    question = question.strip()
    if has_image and "<image>" not in question:
        question = "<image>\n" + question
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer.strip()}]


class ParquetReader:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.pf = pq.ParquetFile(self.path)
        required = {"conversations", "image_bytes"}
        missing = required - set(self.pf.schema_arrow.names)
        if missing:
            raise ValueError("缺少 MindChen-V 必需字段：" + ", ".join(sorted(missing)))
        self.count = self.pf.metadata.num_rows
        self.offsets = [0]
        for g in range(self.pf.num_row_groups):
            self.offsets.append(self.offsets[-1] + self.pf.metadata.row_group(g).num_rows)
        self._conv_cache: OrderedDict[int, Any] = OrderedDict()
        self._image_cache: OrderedDict[int, Any] = OrderedDict()

    def _loc(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        g = bisect.bisect_right(self.offsets, index) - 1
        return g, index - self.offsets[g]

    def _column(self, group: int, key: str):
        cache = self._conv_cache if key == "conversations" else self._image_cache
        if group in cache:
            cache.move_to_end(group)
            return cache[group]
        arr = self.pf.read_row_group(group, columns=[key]).column(key)
        cache[group] = arr
        # Images are potentially large: cache only one image row group.
        limit = 2 if key == "conversations" else 1
        while len(cache) > limit:
            cache.popitem(last=False)
        return arr

    def conversation(self, index: int) -> str:
        group, local = self._loc(index)
        raw = self._column(group, "conversations")[local].as_py()
        return raw if isinstance(raw, str) else dumps_conv(raw)

    def image(self, index: int) -> bytes:
        """Read just a small image batch, rather than whole huge row groups."""
        group, local = self._loc(index)
        batch_size = 64
        batch_no, offset = divmod(local, batch_size)
        key = (group, batch_no)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            batch = self._image_cache[key]
        else:
            import itertools
            batches = self.pf.iter_batches(
                row_groups=[group], batch_size=batch_size, columns=["image_bytes"]
            )
            batch = next(itertools.islice(batches, batch_no, None), None)
            if batch is None:
                raise IndexError(f"图片序号无效: {index}")
            self._image_cache[key] = batch
            while len(self._image_cache) > 2:
                self._image_cache.popitem(last=False)
        raw = batch.column(0)[offset].as_py()
        return image_as_bytes(raw) if raw is not None else BLACK_IMAGE


@dataclass
class Record:
    index: int
    kind: str  # source / added
    turns: list[dict]
    image_path: str | None
    deleted: bool
    changed: bool
    added_id: int | None = None


class ProjectStore:
    """UI-thread controller. Worker threads use their own database connection."""

    def __init__(self, project_path: str, create: bool = False):
        self.path = os.path.abspath(project_path)
        if not create and not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS overrides(
            src_index INTEGER PRIMARY KEY,
            conversations TEXT,
            image_path TEXT,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS additions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            conversations TEXT NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        """)
        self.db.commit()
        self.parquet: ParquetReader | None = None
        self._source_path = self.meta("source_path")
        if self._source_path:
            if not os.path.isfile(self._source_path):
                raise FileNotFoundError(f"原始 Parquet 文件不存在：{self._source_path}")
            self.parquet = ParquetReader(self._source_path)
        self.refresh_added()

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()

    def meta(self, name: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (name,)).fetchone()
        return row[0] if row else None

    def set_meta(self, name: str, value: str):
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (name, value))
        self.db.commit()

    def attach_parquet(self, path: str):
        if self._source_path:
            raise ValueError("当前项目已经绑定 Parquet，请新建项目导入另一份数据")
        reader = ParquetReader(path)
        self.parquet = reader
        self._source_path = reader.path
        self.set_meta("source_path", reader.path)

    def refresh_added(self):
        self.added_ids = [int(r[0]) for r in self.db.execute("SELECT id FROM additions ORDER BY id")]

    @property
    def source_count(self) -> int:
        return self.parquet.count if self.parquet else 0

    @property
    def count(self) -> int:
        return self.source_count + len(self.added_ids)

    def _added_id(self, index: int) -> int:
        return self.added_ids[index - self.source_count]

    def _override(self, index: int):
        return self.db.execute(
            "SELECT conversations,image_path,deleted FROM overrides WHERE src_index=?",
            (index,),
        ).fetchone()

    def record(self, index: int) -> Record:
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index < self.source_count:
            ov = self._override(index)
            conv = ov[0] if ov and ov[0] is not None else self.parquet.conversation(index)
            return Record(index, "source", parses_conv(conv), ov[1] if ov else None,
                          bool(ov[2]) if ov else False, bool(ov), None)
        id_ = self._added_id(index)
        row = self.db.execute(
            "SELECT image_path,conversations,deleted FROM additions WHERE id=?", (id_,)
        ).fetchone()
        return Record(index, "added", parses_conv(row[1]), row[0], bool(row[2]), True, id_)

    def image(self, record: Record) -> bytes:
        if record.image_path:
            with open(record.image_path, "rb") as f:
                return f.read()
        if record.kind == "source":
            return self.parquet.image(record.index)
        return BLACK_IMAGE

    def summary(self, index: int) -> tuple[str, str, str]:
        if index < self.source_count:
            row = self._override(index)
            if row and row[0] is not None:
                conv = row[0]
            else:
                conv = self.parquet.conversation(index)
            typ = "Parquet"
            status = "已删除" if row and row[2] else ("已修改" if row else "原始")
        else:
            id_ = self._added_id(index)
            row = self.db.execute(
                "SELECT conversations,deleted FROM additions WHERE id=?", (id_,)
            ).fetchone()
            conv = row[0]
            typ = "新增"
            status = "已删除" if row[1] else "已添加"
        try:
            turns = parses_conv(conv)
            prompt = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
            prompt = str(prompt).replace("<image>", "[图]").replace("\n", " ").strip()
            if any(not str(t.get("content", "")).strip() for t in turns):
                status = "待标注" if status != "已删除" else status
            return typ, status, prompt[:200]
        except Exception:
            return typ, "格式错误", "<无法解析 conversations>"

    def save(self, rec: Record, turns: list[dict], image_path: str | None):
        val = dumps_conv(turns)
        if rec.kind == "source":
            self.db.execute(
                """INSERT INTO overrides(src_index,conversations,image_path,deleted)
                   VALUES(?,?,?,?) ON CONFLICT(src_index) DO UPDATE SET
                   conversations=excluded.conversations,image_path=excluded.image_path,
                   deleted=excluded.deleted""",
                (rec.index, val, image_path, int(rec.deleted)),
            )
        else:
            self.db.execute(
                "UPDATE additions SET conversations=?,image_path=? WHERE id=?",
                (val, image_path, rec.added_id),
            )
        self.db.commit()

    def mark_deleted(self, index: int, deleted: bool):
        if index < self.source_count:
            self.db.execute(
                """INSERT INTO overrides(src_index,deleted) VALUES(?,?)
                   ON CONFLICT(src_index) DO UPDATE SET deleted=excluded.deleted""",
                (index, int(deleted)),
            )
        else:
            self.db.execute(
                "UPDATE additions SET deleted=? WHERE id=?",
                (int(deleted), self._added_id(index)),
            )
        self.db.commit()

    def add_text(self, question: str, answer: str) -> int:
        turns = dumps_conv(make_turns(question, answer, False))
        cur = self.db.execute(
            "INSERT INTO additions(image_path,conversations) VALUES(NULL,?)", (turns,)
        )
        self.db.commit()
        self.added_ids.append(cur.lastrowid)
        return self.count - 1

    def stats(self) -> tuple[int, int]:
        src_del = self.db.execute("SELECT count(*) FROM overrides WHERE deleted=1").fetchone()[0]
        add_del = self.db.execute("SELECT count(*) FROM additions WHERE deleted=1").fetchone()[0]
        return self.count, self.count - src_del - add_del

# ====== 后台导入与导出模块 ======
SCHEMA = pa.schema([
    pa.field("conversations", pa.string()),
    pa.field("image_bytes", pa.binary()),
])


def normalize_image_bytes(data: bytes, size: int, quality: int) -> bytes:
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        if size > 0:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        result = io.BytesIO()
        image.save(result, format="JPEG", quality=quality, optimize=True)
        return result.getvalue()


class ImportWorker(QThread):
    progress = Signal(int, int)  # read, added
    result = Signal(str)
    failure = Signal(str)

    def __init__(self, project: str, mode: str, path: str,
                 question: str = "请描述这张图片。", answer: str = "",
                 recursive: bool = True):
        super().__init__()
        self.project, self.mode, self.path = project, mode, path
        self.question, self.answer, self.recursive = question, answer, recursive
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def _folder_rows(self):
        ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if self.recursive:
            for root, dirs, files in os.walk(self.path):
                dirs.sort()
                for name in sorted(files):
                    if self.cancelled:
                        return
                    if Path(name).suffix.lower() in ext:
                        yield os.path.abspath(os.path.join(root, name)), self.question, self.answer
        else:
            for name in sorted(os.listdir(self.path)):
                if self.cancelled:
                    return
                file = os.path.join(self.path, name)
                if os.path.isfile(file) and Path(name).suffix.lower() in ext:
                    yield os.path.abspath(file), self.question, self.answer

    def _csv_rows(self):
        base = Path(self.path).resolve().parent
        with open(self.path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not ({"image", "image_path"} & set(reader.fieldnames)):
                raise ValueError("CSV 必须有 image 或 image_path 列")
            if not {"question", "answer"}.issubset(set(reader.fieldnames)):
                raise ValueError("CSV 必须有 question 和 answer 列")
            for row in reader:
                if self.cancelled:
                    return
                raw = (row.get("image_path") or row.get("image") or "").strip()
                img = Path(raw)
                if not img.is_absolute():
                    img = base / img
                yield str(img.resolve()) if raw else None, (row.get("question") or ""), (row.get("answer") or "")

    def run(self):
        db = None
        try:
            db = sqlite3.connect(self.project, timeout=60)
            db.execute("PRAGMA busy_timeout=60000")
            rows = self._folder_rows() if self.mode == "folder" else self._csv_rows()
            added, read, skipped = 0, 0, 0
            buf = []
            for image, question, answer in rows:
                if self.cancelled:
                    break
                read += 1
                if not image or not os.path.isfile(image):
                    skipped += 1
                    continue
                buf.append((image, dumps_conv(make_turns(question, answer, True))))
                if len(buf) >= 300:
                    db.executemany("INSERT INTO additions(image_path,conversations) VALUES(?,?)", buf)
                    db.commit()
                    added += len(buf)
                    buf.clear()
                    self.progress.emit(read, added)
            if buf:
                db.executemany("INSERT INTO additions(image_path,conversations) VALUES(?,?)", buf)
                db.commit()
                added += len(buf)
            self.progress.emit(read, added)
            status = "已取消" if self.cancelled else "导入完成"
            self.result.emit(f"{status}：读取 {read} 项，新添加 {added} 项，找不到图片 {skipped} 项。")
        except Exception as e:
            self.failure.emit(f"导入失败：{e}\n\n{traceback.format_exc()}")
        finally:
            if db:
                db.close()


class ExportWorker(QThread):
    progress = Signal(int, int)  # processed, total
    result = Signal(str)
    failure = Signal(str)

    def __init__(self, project: str, target: str, image_size: int = 256,
                 jpeg_quality: int = 95, compression: str = "zstd",
                 keep_original: bool = True, batch_size: int = 256):
        super().__init__()
        self.project, self.target = project, os.path.abspath(target)
        self.image_size, self.jpeg_quality = image_size, jpeg_quality
        self.compression, self.keep_original = compression, keep_original
        self.batch_size = batch_size
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def _prepare(self, turns_raw, image_raw, image_path, index):
        turns = parses_conv(turns_raw)
        issues = validate_turns(turns)
        if issues:
            raise ValueError(f"第 {index + 1} 条数据无效：" + "; ".join(issues))
        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"第 {index + 1} 条引用图片不存在：{image_path}")
            image = encode_image(image_path, self.image_size, self.jpeg_quality)
        elif image_raw is None:
            image = BLACK_IMAGE
        else:
            image = image_as_bytes(image_raw)
            if not self.keep_original:
                image = normalize_image_bytes(image, self.image_size, self.jpeg_quality)
        # Preserve the turn objects including optional fields (tools/etc.) from source.
        return {"conversations": dumps_conv(turns), "image_bytes": image}

    def run(self):
        store = None
        writer = None
        tmp = self.target + ".part"
        try:
            store = ProjectStore(self.project)
            if store.parquet and os.path.abspath(store.parquet.path) == self.target:
                raise ValueError("不能覆盖正在使用的源 Parquet，请另存到不同位置")
            os.makedirs(os.path.dirname(self.target), exist_ok=True)
            total, active = store.stats()
            writer = pq.ParquetWriter(
                tmp, SCHEMA, compression=self.compression, use_dictionary=["conversations"]
            )
            done, exported = 0, 0
            pending = []

            def flush():
                nonlocal exported
                if pending:
                    table = pa.Table.from_pylist(pending, schema=SCHEMA)
                    writer.write_table(table, row_group_size=len(pending))
                    exported += len(pending)
                    pending.clear()

            if store.parquet:
                # Streaming batches, NO global dataset materialization.
                for batch in store.parquet.pf.iter_batches(
                    batch_size=self.batch_size, columns=["conversations", "image_bytes"]
                ):
                    if self.cancelled:
                        break
                    n = batch.num_rows
                    overrides = {
                        int(r[0]): r[1:]
                        for r in store.db.execute(
                            "SELECT src_index,conversations,image_path,deleted FROM overrides "
                            "WHERE src_index>=? AND src_index<?", (done, done + n)
                        )
                    }
                    convs = batch.column(0)
                    imgs = batch.column(1)
                    for j in range(n):
                        if self.cancelled:
                            break
                        idx = done + j
                        ov = overrides.get(idx)
                        if ov and ov[2]:
                            continue
                        source_conv = convs[j].as_py()
                        conv = ov[0] if ov and ov[0] is not None else source_conv
                        path = ov[1] if ov else None
                        # Only read/decode image data for a row that is not deleted.
                        raw_img = None if path else imgs[j].as_py()
                        pending.append(self._prepare(conv, raw_img, path, idx))
                        if len(pending) >= self.batch_size:
                            flush()
                    done += n
                    self.progress.emit(done, total)

            if not self.cancelled:
                cursor = store.db.execute(
                    "SELECT image_path,conversations,deleted FROM additions ORDER BY id"
                )
                for image, conv, deleted in cursor:
                    if self.cancelled:
                        break
                    if not deleted:
                        pending.append(self._prepare(conv, None, image, done))
                        if len(pending) >= self.batch_size:
                            flush()
                    done += 1
                    if done % self.batch_size == 0:
                        self.progress.emit(done, total)
            if not self.cancelled:
                flush()
            writer.close()
            writer = None
            if self.cancelled:
                if os.path.exists(tmp):
                    os.remove(tmp)
                self.result.emit("已取消导出，未生成不完整数据集。")
            else:
                os.replace(tmp, self.target)
                self.progress.emit(total, total)
                self.result.emit(f"导出成功：{self.target}\n共 {exported} 条有效样本（项目中未删除 {active} 条）。")
        except Exception as e:
            self.failure.emit(f"导出失败：{e}\n\n{traceback.format_exc()}")
        finally:
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if store:
                store.close()

# ====== 桌面 GUI 模块 ======
APP_NAME = "Multimodal Dataset Studio"

DARK_STYLE = """
QWidget { background-color: #181c25; color: #e9eef7; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 12px; }
QMainWindow, QDialog { background: #181c25; }
QMenuBar, QMenu, QToolBar { background: #222734; color: #e9eef7; }
QToolBar { border: 0px; border-bottom: 1px solid #364050; spacing: 5px; padding: 5px; }
QToolButton { background: #303949; border: 1px solid #3c485c; border-radius: 6px; padding: 7px 10px; }
QToolButton:hover { background: #394966; }
QPushButton { background: #303a4b; border: 1px solid #4a556a; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #41516a; }
QPushButton:pressed { background: #526786; }
QPushButton:disabled { background: #222733; color: #6d7485; }
QLineEdit, QPlainTextEdit, QSpinBox, QComboBox { background: #232937; color: #f1f5fb; border: 1px solid #455268; border-radius: 6px; padding: 5px; selection-background-color: #3766ab; }
QLineEdit:focus, QPlainTextEdit:focus { border: 1px solid #599afa; }
QTableView, QTableWidget { background: #202634; alternate-background-color: #252d3c; border: 1px solid #39475c; gridline-color: #303b4b; selection-background-color: #344f7b; selection-color: white; }
QHeaderView::section { background: #2c3443; padding: 7px; border: 0; border-right: 1px solid #3f4859; color: #cad9eb; }
QScrollBar:vertical, QScrollBar:horizontal { background: #1e2532; width: 13px; height: 13px; }
QScrollBar::handle { background: #57647a; border-radius: 5px; min-height: 22px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QGroupBox { border: 1px solid #39465a; border-radius: 8px; margin-top: 11px; padding-top: 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QStatusBar { background: #232b38; color: #cbd5e6; }
QLabel#preview { background: #111721; border: 1px solid #354153; border-radius: 8px; }
QLabel#hint { color: #aab8cb; }
QSplitter::handle { background: #333c4d; }
"""


class DatasetModel(QAbstractTableModel):
    HEADERS = ["序号", "来源", "状态", "用户问题（摘要）"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.store: ProjectStore | None = None
        self._cache: OrderedDict[int, tuple] = OrderedDict()

    def reset_store(self, store):
        self.beginResetModel()
        self.store = store
        self._cache.clear()
        self.endResetModel()

    def refresh_rows(self):
        self.beginResetModel()
        self._cache.clear()
        self.endResetModel()

    def refresh_one(self, row):
        self._cache.pop(row, None)
        if self.store and row < self.store.count:
            self.dataChanged.emit(self.index(row, 0), self.index(row, 3))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or not self.store else self.store.count

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not self.store:
            return None
        row = index.row()
        if role in (Qt.DisplayRole, Qt.ForegroundRole):
            try:
                if row not in self._cache:
                    self._cache[row] = self.store.summary(row)
                    while len(self._cache) > 1500:
                        self._cache.popitem(last=False)
                typ, status, preview = self._cache[row]
            except Exception as exc:
                typ, status, preview = "错误", "读取失败", str(exc)
            if role == Qt.ForegroundRole:
                if status == "已删除":
                    return QColor("#8d95a5")
                if status in ("待标注", "格式错误", "读取失败"):
                    return QColor("#f5be76")
                if status == "已修改":
                    return QColor("#8bdab6")
                return None
            return [str(row + 1), typ, status, preview][index.column()]
        if role == Qt.TextAlignmentRole and index.column() == 0:
            return Qt.AlignCenter
        return None


class PreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(240)
        self.setMinimumWidth(260)
        self._original: QPixmap | None = None
        self.setText("请选择数据查看图片")

    def show_bytes(self, data: bytes):
        pix = QPixmap()
        try:
            if not pix.loadFromData(data):
                with Image.open(io.BytesIO(data)) as img:
                    preview = ImageOps.exif_transpose(img).convert("RGB")
                    buf = io.BytesIO()
                    preview.save(buf, "PNG")
                    if not pix.loadFromData(buf.getvalue()):
                        raise ValueError("Qt 无法解析图片")
            self._original = pix
            self._refresh()
        except Exception as e:
            self._original = None
            self.setText(f"图片预览失败：\n{e}")

    def reset_preview(self):
        self._original = None
        self.setText("请选择数据查看图片")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._original and not self._original.isNull():
            sz = self.size() - QSize(14, 14)
            if sz.width() > 0 and sz.height() > 0:
                self.setPixmap(self._original.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片导入设置")
        self.resize(470, 235)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.question = QLineEdit("请描述图片中的内容。")
        self.answer = QLineEdit("")
        self.answer.setPlaceholderText("留空表示待标注（导出前必须填写）")
        self.recursive = QCheckBox("包含所有子文件夹")
        self.recursive.setChecked(True)
        form.addRow("默认用户提问：", self.question)
        form.addRow("默认标准回答：", self.answer)
        form.addRow("遍历方式：", self.recursive)
        v.addLayout(form)
        tip = QLabel("每张图片新建一条问答；自动向用户提问添加 <image> 标记。")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        v.addWidget(tip)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出 MindChen-V SFT Parquet")
        self.resize(520, 270)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.size = QSpinBox()
        self.size.setRange(0, 2048)
        self.size.setSingleStep(32)
        self.size.setValue(256)
        self.size.setSuffix(" px (0=不缩放)")
        self.quality = QSpinBox()
        self.quality.setRange(70, 100)
        self.quality.setValue(95)
        self.compression = QComboBox()
        self.compression.addItems(["zstd", "snappy"])
        self.keep_original = QCheckBox("源 Parquet 未替换的图片直接复制二进制（快速且无损）")
        self.keep_original.setChecked(True)
        form.addRow("新增 / 替换图片尺寸：", self.size)
        form.addRow("JPEG 质量：", self.quality)
        form.addRow("Parquet 压缩：", self.compression)
        v.addLayout(form)
        v.addWidget(self.keep_original)
        note = QLabel("MindChen-V 官方示例通常使用 256×256 图片。勾选源图直拷可避免数百万张重复编码；取消勾选则所有图片统一重编码。")
        note.setObjectName("hint")
        note.setWordWrap(True)
        v.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME + " — 未打开项目")
        self.resize(1450, 880)
        self.store: ProjectStore | None = None
        self.current_index: int | None = None
        self.current_record: Record | None = None
        self.current_image_path: str | None = None
        self.dirty = False
        self.loading = False
        self.worker = None
        self.progress_dialog = None
        self.model = DatasetModel(self)
        self._build_ui()
        self._update_stats("请先新建或打开一个 .sftproj 项目")

    def _action(self, toolbar, text, callback, shortcut=None):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(callback)
        toolbar.addAction(act)
        return act

    def _build_ui(self):
        toolbar = QToolBar("操作", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        self._action(toolbar, "新建项目", self.new_project, "Ctrl+N")
        self._action(toolbar, "打开项目", self.open_project, "Ctrl+O")
        toolbar.addSeparator()
        self._action(toolbar, "导入 Parquet", self.attach_parquet, "Ctrl+Shift+I")
        self._action(toolbar, "导入图片文件夹", self.import_folder, "Ctrl+I")
        self._action(toolbar, "导入问答 CSV", self.import_csv)
        self._action(toolbar, "添加纯文本", self.add_text)
        toolbar.addSeparator()
        self._action(toolbar, "保存当前", self.save_current, "Ctrl+S")
        self._action(toolbar, "检查当前", self.validate_current)
        self._action(toolbar, "导出 Parquet", self.export_parquet, "Ctrl+E")

        root = QWidget()
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(10, 10, 10, 8)
        root_l.setSpacing(8)
        header = QHBoxLayout()
        self.project_label = QLabel("项目：未打开")
        self.project_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.project_label, 1)
        header.addWidget(QLabel("跳到第"))
        self.jump_box = QSpinBox()
        self.jump_box.setRange(1, 1)
        self.jump_box.setFixedWidth(120)
        self.jump_box.setKeyboardTracking(False)
        self.jump_box.editingFinished.connect(self.jump_to_index)
        header.addWidget(self.jump_box)
        header.addWidget(QLabel("条"))
        root_l.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        root_l.addWidget(splitter, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        title = QLabel("数据列表（支持百万行虚拟滚动）")
        title.setStyleSheet("font-weight: 600; font-size: 14px")
        left_l.addWidget(title)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 64)
        self.table.setColumnWidth(1, 75)
        self.table.setColumnWidth(2, 75)
        self.table.selectionModel().currentRowChanged.connect(self.on_row_changed)
        left_l.addWidget(self.table, 1)
        controls = QHBoxLayout()
        for caption, fn in [("← 上一条", self.prev_row), ("下一条 →", self.next_row),
                            ("删除 / 恢复", self.toggle_delete)]:
            btn = QPushButton(caption)
            btn.clicked.connect(fn)
            controls.addWidget(btn)
        left_l.addLayout(controls)
        splitter.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(9)
        preview_group = QGroupBox("图片预览")
        preview_l = QVBoxLayout(preview_group)
        self.preview = PreviewLabel()
        preview_l.addWidget(self.preview, 1)
        img_bar = QHBoxLayout()
        self.image_path_label = QLabel("来源图片：—")
        self.image_path_label.setObjectName("hint")
        self.image_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        img_bar.addWidget(self.image_path_label, 1)
        self.replace_btn = QPushButton("替换图片")
        self.replace_btn.clicked.connect(self.replace_image)
        img_bar.addWidget(self.replace_btn)
        self.restore_btn = QPushButton("恢复原图")
        self.restore_btn.clicked.connect(self.restore_image)
        img_bar.addWidget(self.restore_btn)
        preview_l.addLayout(img_bar)
        right_l.addWidget(preview_group, 5)

        conv_group = QGroupBox("多轮对话编辑")
        conv_l = QVBoxLayout(conv_group)
        self.turn_table = QTableWidget(0, 2)
        self.turn_table.setHorizontalHeaderLabels(["角色", "内容（可编辑）"])
        self.turn_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.turn_table.setColumnWidth(0, 110)
        self.turn_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.turn_table.verticalHeader().setVisible(False)
        self.turn_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.turn_table.setSelectionMode(QAbstractItemView.SingleSelection)
        conv_l.addWidget(self.turn_table, 1)
        edit_bar = QHBoxLayout()
        for name, fn in [("+ 用户", lambda: self.insert_turn("user")),
                         ("+ 助手", lambda: self.insert_turn("assistant")),
                         ("+ 系统", lambda: self.insert_turn("system")),
                         ("删除轮次", self.remove_turn), ("↑", self.move_turn_up),
                         ("↓", self.move_turn_down)]:
            btn = QPushButton(name)
            btn.clicked.connect(fn)
            edit_bar.addWidget(btn)
        conv_l.addLayout(edit_bar)
        bottom = QHBoxLayout()
        self.note_label = QLabel("提示：图片问答的用户消息应包含 <image> 占位符。")
        self.note_label.setObjectName("hint")
        bottom.addWidget(self.note_label, 1)
        self.save_btn = QPushButton("保存当前修改 (Ctrl+S)")
        self.save_btn.setStyleSheet("background:#285d9f; border:1px solid #4082cb; font-weight:600")
        self.save_btn.clicked.connect(self.save_current)
        bottom.addWidget(self.save_btn)
        conv_l.addLayout(bottom)
        right_l.addWidget(conv_group, 6)
        splitter.addWidget(right)
        splitter.setSizes([570, 870])

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar(self))

    def _ensure_project(self) -> bool:
        if self.store:
            return True
        QMessageBox.information(self, "请新建项目", "请先点击“新建项目”，选择 .sftproj 项目保存位置。")
        return False

    def _swap_project(self, path, create=False):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "任务进行中", "请先完成或取消当前导入/导出任务。")
            return
        if not self._save_if_needed():
            return
        try:
            new_store = ProjectStore(path, create=create)
        except Exception as e:
            QMessageBox.critical(self, "打开项目失败", str(e))
            return
        if self.store:
            self.store.close()
        self.store = new_store
        self.current_record = None
        self.current_index = None
        self.current_image_path = None
        self.dirty = False
        self.preview.reset_preview()
        self.turn_table.setRowCount(0)
        self.model.reset_store(self.store)
        self.setWindowTitle(f"{APP_NAME} — {Path(path).name}")
        self.project_label.setText(f"项目：{path}")
        self.jump_box.setMaximum(max(self.store.count, 1))
        self._update_stats("项目已打开")
        if self.store.count:
            self.select_index(0)

    def new_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "新建项目", "my_dataset.sftproj", "数据集项目 (*.sftproj)")
        if not path:
            return
        if not path.lower().endswith(".sftproj"):
            path += ".sftproj"
        if os.path.exists(path):
            if QMessageBox.question(self, "项目已存在", "文件已存在，是否直接打开（不会清除已有数据）？") != QMessageBox.Yes:
                return
            self._swap_project(path)
        else:
            self._swap_project(path, create=True)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "数据集项目 (*.sftproj)")
        if path:
            self._swap_project(path)

    def attach_parquet(self):
        if not self._ensure_project():
            return
        if not self._save_if_needed():
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择 MindChen-V Parquet", "", "Parquet (*.parquet)")
        if not path:
            return
        try:
            self.store.attach_parquet(path)
            self.current_index = None
            self.current_record = None
            self.current_image_path = None
            self.turn_table.setRowCount(0)
            self.preview.reset_preview()
            self.model.refresh_rows()
            self.jump_box.setMaximum(max(self.store.count, 1))
            self._update_stats(f"已绑定 Parquet：{Path(path).name}")
            self.select_index(0)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def import_folder(self):
        if not self._ensure_project() or not self._save_if_needed():
            return
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not path:
            return
        dlg = ImportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._start_import("folder", path, dlg.question.text(), dlg.answer.text(), dlg.recursive.isChecked())

    def import_csv(self):
        if not self._ensure_project() or not self._save_if_needed():
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择带 question / answer 的 CSV", "", "CSV (*.csv)")
        if path:
            self._start_import("csv", path)

    def _start_import(self, mode, path, question="", answer="", recursive=True):
        self._disable_data_actions(True)
        self.worker = ImportWorker(self.store.path, mode, path, question, answer, recursive)
        self.progress_dialog = QProgressDialog("正在分批导入，项目自动保存…", "取消", 0, 0, self)
        self.progress_dialog.setWindowTitle("导入")
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.worker.progress.connect(lambda read, added: self.progress_dialog.setLabelText(
            f"已扫描 {read:,} 项；导入 {added:,} 条（可取消）"))
        self.worker.result.connect(self._worker_ok)
        self.worker.failure.connect(self._worker_error)
        self.worker.finished.connect(self._import_finished)
        self.worker.start()

    def _import_finished(self):
        self._disable_data_actions(False)
        self.progress_dialog.close()
        self.store.refresh_added()
        self.model.refresh_rows()
        self.jump_box.setMaximum(max(self.store.count, 1))
        self._update_stats("导入任务结束")
        if self.current_index is None and self.store.count:
            self.select_index(0)

    def add_text(self):
        if not self._ensure_project() or not self._save_if_needed():
            return
        idx = self.store.add_text("请输入纯文本问题", "")
        self.model.refresh_rows()
        self.jump_box.setMaximum(max(self.store.count, 1))
        self._update_stats("新增纯文本问答（导出时生成黑图占位）")
        self.select_index(idx)

    def _save_if_needed(self):
        if self.dirty:
            return self.save_current()
        return True

    def on_row_changed(self, current, previous):
        if not current.isValid() or not self.store:
            return
        idx = current.row()
        if idx == self.current_index:
            return
        if not self._save_if_needed():
            if self.current_index is not None:
                self.select_index(self.current_index)
            return
        self._load_record(idx)

    def select_index(self, idx: int):
        if not self.store or not 0 <= idx < self.store.count:
            return
        self.table.selectRow(idx)
        self.table.setCurrentIndex(self.model.index(idx, 0))
        self.table.scrollTo(self.model.index(idx, 0), QAbstractItemView.PositionAtCenter)

    def jump_to_index(self):
        if self.store and self.store.count:
            self.select_index(self.jump_box.value() - 1)

    def prev_row(self):
        if self.current_index is not None:
            self.select_index(max(0, self.current_index - 1))

    def next_row(self):
        if self.store and self.current_index is not None:
            self.select_index(min(self.store.count - 1, self.current_index + 1))

    def _load_record(self, idx):
        try:
            rec = self.store.record(idx)
            img = self.store.image(rec)
        except Exception as e:
            QMessageBox.warning(self, "无法读取该条数据", f"序号 {idx + 1}:\n{e}")
            return
        self.loading = True
        self.current_record = rec
        self.current_index = idx
        self.current_image_path = rec.image_path
        self.preview.show_bytes(img)
        desc = rec.image_path if rec.image_path else ("源 Parquet 内嵌图片" if rec.kind == "source" else "纯文本黑图占位")
        self.image_path_label.setText(desc)
        self.image_path_label.setToolTip(desc)
        self._set_turns(rec.turns)
        self.dirty = False
        self.jump_box.blockSignals(True)
        self.jump_box.setValue(idx + 1)
        self.jump_box.blockSignals(False)
        self.note_label.setText("已删除（导出时跳过）" if rec.deleted else "编辑问答后点击保存；快捷键 Ctrl+S")
        self.loading = False
        self._update_stats(f"当前第 {idx + 1:,} 条 / {self.store.count:,} 条")

    def _set_turns(self, turns):
        self.loading = True
        self.turn_table.setRowCount(0)
        for t in turns:
            self._append_turn(str(t.get("role", "user")), str(t.get("content", "")), t)
        self.loading = False

    def _append_turn(self, role, content, extra=None):
        row = self.turn_table.rowCount()
        self.turn_table.insertRow(row)
        self.turn_table.setRowHeight(row, 94)
        combo = QComboBox()
        combo.addItems(["user", "assistant", "system"])
        if role in ["user", "assistant", "system"]:
            combo.setCurrentText(role)
        combo.currentTextChanged.connect(self._mark_dirty)
        self.turn_table.setCellWidget(row, 0, combo)
        editor = QPlainTextEdit()
        editor.setPlaceholderText("填写消息内容…")
        editor.setPlainText(content)
        editor.textChanged.connect(self._mark_dirty)
        self.turn_table.setCellWidget(row, 1, editor)
        # Preserve ancillary conversation keys when reading existing Parquet.
        combo.setProperty("extra", dict(extra or {}))

    def _get_turns(self):
        result = []
        for row in range(self.turn_table.rowCount()):
            combo = self.turn_table.cellWidget(row, 0)
            edit = self.turn_table.cellWidget(row, 1)
            obj = dict(combo.property("extra") or {})
            obj["role"] = combo.currentText()
            obj["content"] = edit.toPlainText()
            result.append(obj)
        return result

    def _mark_dirty(self, *args):
        if not self.loading:
            self.dirty = True
            self.note_label.setText("● 有尚未保存的修改")

    def insert_turn(self, role):
        if self.current_record is None:
            return
        self._append_turn(role, "")
        self.turn_table.selectRow(self.turn_table.rowCount() - 1)
        self._mark_dirty()

    def remove_turn(self):
        row = self.turn_table.currentRow()
        if row >= 0:
            self.turn_table.removeRow(row)
            self._mark_dirty()

    def _move_turn(self, direction):
        row = self.turn_table.currentRow()
        new = row + direction
        if row < 0 or not 0 <= new < self.turn_table.rowCount():
            return
        turns = self._get_turns()
        turns[row], turns[new] = turns[new], turns[row]
        self._set_turns(turns)
        self.turn_table.selectRow(new)
        self._mark_dirty()

    def move_turn_up(self):
        self._move_turn(-1)

    def move_turn_down(self):
        self._move_turn(1)

    def save_current(self):
        if not self.store or not self.current_record:
            return True
        if not self.dirty:
            return True
        turns = self._get_turns()
        # Edit/save may be partial drafts; full content checks run when exporting.
        if not turns:
            QMessageBox.warning(self, "无法保存", "至少需要一轮对话。")
            return False
        try:
            self.store.save(self.current_record, turns, self.current_image_path)
            self.current_record = self.store.record(self.current_index)
            self.dirty = False
            self.note_label.setText("已保存。导出时会检查空白答案或不合法对话。")
            self.model.refresh_one(self.current_index)
            self._update_stats("保存成功")
            return True
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return False

    def replace_image(self):
        if not self.current_record:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择替换图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not path:
            return
        try:
            data = Path(path).read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            self.current_image_path = os.path.abspath(path)
            self.preview.show_bytes(data)
            self.image_path_label.setText(self.current_image_path)
            self._mark_dirty()
        except Exception as e:
            QMessageBox.warning(self, "图片无效", str(e))

    def restore_image(self):
        if not self.current_record:
            return
        if self.current_record.kind == "added":
            QMessageBox.information(self, "提示", "新增样本无法恢复源图片。纯文本样本可继续替换为图片。")
            return
        try:
            self.current_image_path = None
            self.preview.show_bytes(self.store.parquet.image(self.current_record.index))
            self.image_path_label.setText("源 Parquet 内嵌图片")
            self._mark_dirty()
        except Exception as e:
            QMessageBox.critical(self, "恢复失败", str(e))

    def toggle_delete(self):
        if not self.current_record or not self._save_if_needed():
            return
        new_state = not self.current_record.deleted
        self.store.mark_deleted(self.current_index, new_state)
        self.current_record.deleted = new_state
        self.model.refresh_one(self.current_index)
        self.note_label.setText("已删除（导出时跳过）" if new_state else "已恢复")
        self._update_stats("已标记删除" if new_state else "已恢复记录")

    def validate_current(self):
        if not self.current_record:
            return
        turns = self._get_turns()
        issues = validate_turns(turns)
        if self.current_image_path and not os.path.isfile(self.current_image_path):
            issues.append("引用的图片文件不存在")
        if not issues and self.current_image_path and not any(
            t.get("role") == "user" and "<image>" in str(t.get("content", "")) for t in turns
        ):
            issues.append("提示：用户问题里没有 <image> 视觉占位符（可能无法按视觉任务训练）")
        if issues:
            QMessageBox.warning(self, "数据检查", "\n".join(issues))
        else:
            QMessageBox.information(self, "数据检查", "当前对话结构与非空字段检查通过。")

    def export_parquet(self):
        if not self._ensure_project() or not self._save_if_needed():
            return
        if not self.store.count:
            QMessageBox.information(self, "没有数据", "请先导入图片或 Parquet。")
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        target, _ = QFileDialog.getSaveFileName(self, "保存训练数据", "my_sft_i2t.parquet", "Parquet (*.parquet)")
        if not target:
            return
        if not target.lower().endswith(".parquet"):
            target += ".parquet"
        if self.store.parquet and os.path.abspath(target) == os.path.abspath(self.store.parquet.path):
            QMessageBox.warning(self, "不允许覆盖", "请将数据集另存为新文件，不能覆盖当前使用的源 Parquet。")
            return
        if os.path.exists(target) and QMessageBox.question(self, "覆盖文件", "目标文件已经存在，确定覆盖？") != QMessageBox.Yes:
            return
        self._disable_data_actions(True)
        self.worker = ExportWorker(self.store.path, target, dlg.size.value(),
                                   dlg.quality.value(), dlg.compression.currentText(),
                                   dlg.keep_original.isChecked())
        self.progress_dialog = QProgressDialog("正在分批写入 Parquet…", "取消导出", 0,
                                                min(self.store.count, 2_147_483_647), self)
        self.progress_dialog.setWindowTitle("导出")
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.worker.progress.connect(self._export_progress)
        self.worker.result.connect(self._worker_ok)
        self.worker.failure.connect(self._worker_error)
        self.worker.finished.connect(self._export_finished)
        self.worker.start()

    def _export_progress(self, done, total):
        if self.progress_dialog:
            self.progress_dialog.setValue(min(done, 2_147_483_647))
            self.progress_dialog.setLabelText(f"已处理 {done:,} / {total:,} 条")

    def _export_finished(self):
        self._disable_data_actions(False)
        if self.progress_dialog:
            self.progress_dialog.close()
        self._update_stats("导出任务结束")

    def _worker_ok(self, info):
        self.statusBar().showMessage(info.replace("\n", " "), 20000)
        QMessageBox.information(self, "任务结果", info)

    def _worker_error(self, info):
        self.statusBar().showMessage(info.split("\n", 1)[0], 20000)
        QMessageBox.critical(self, "任务失败", info[:10000])

    def _disable_data_actions(self, busy):
        self.centralWidget().setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.turn_table.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)
        self.replace_btn.setEnabled(not busy)
        self.restore_btn.setEnabled(not busy)
        # ToolBar QAction operations are blocked while background DB writes run.
        for toolbar in self.findChildren(QToolBar):
            toolbar.setEnabled(not busy)

    def _update_stats(self, msg=""):
        if self.store:
            total, active = self.store.stats()
            self.statusBar().showMessage(f"{msg}    |    总数 {total:,}    |    未删除 {active:,}    |    "
                                             f"Parquet {self.store.source_count:,}    |    新增 {len(self.store.added_ids):,}")
        else:
            self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "任务仍在执行", "请在进度窗口中取消或完成当前任务后关闭软件。")
            event.ignore()
            return
        if not self._save_if_needed():
            event.ignore()
            return
        if self.store:
            self.store.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Multimodal Dataset Studio (单文件版)")
        print("安装: python -m pip install PySide6 pyarrow pillow")
        print("启动: python 多模态数据集编辑器.py")
        print("功能: 导入 Parquet/图片/CSV, 编辑 SFT 问答, SQLite 项目保存, 导出 Parquet")
    else:
        main()
