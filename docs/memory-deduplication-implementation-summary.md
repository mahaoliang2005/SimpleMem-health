# SimpleMem Memory Deduplication 实现改动总结

> 日期: 2026-06-11
> 涉及 Commit: `b52e4e0` .. `13c38e0` (共 9 个 commit)
> 设计文档: `docs/superpowers/specs/2026-06-11-memory-dedup-design.md`
> 实现计划: `docs/superpowers/plans/2026-06-11-memory-dedup.md`

---

## 一、概述

本次实现为 SimpleMem 增加了**写入时向量相似度去重**（inline deduplication）能力。核心思路是：在将 `MemoryEntry` 写入向量存储前，计算批次内条目的 pairwise 余弦相似度，对相似度超过阈值的近_duplicate_对，用 `superseded_by` 字段进行软删除标记，只将存活条目写入数据库。

**关键设计约束：**
- **零 LLM 调用**：去重纯靠向量相似度计算，不引入额外的 LLM 推理开销
- **Fail-open**：任何 embedding / 相似度计算失败都原样返回，不阻断写入流水线
- **软删除**：`superseded_by` 标记而非物理删除，兼容现有 Cross 架构
- **并行处理无损**：Worker 线程逻辑完全不变，去重发生在主线程的统一 batch 上
- **传递性链处理**：多条目互相重复时，确保所有 loser 的 `superseded_by` 指向 root winner

---

## 二、逐文件改动详情

### 2.1 `models/memory_entry.py`

**Commit:** `b52e4e0`

| 行号 | 改动 |
|------|------|
| 52-55 | **新增字段** `superseded_by: Optional[str] = Field(None, description="ID of the entry that superseded this one (soft delete marker)")`。该字段位于 `topic` 字段之后，作为 MemoryEntry 的软删除标记。 |
| 69 | **更新示例**：`json_schema_extra.example` 中增加 `"superseded_by": null`，使 Pydantic 示例与 schema 一致。 |

**动机：** 为 MemoryEntry 引入软删除标记，这是整个去重系统的数据基础。`superseded_by` 存储 winner 的 `entry_id`，为空表示该条目未被替代。

---

### 2.2 `core/vector_dedup.py`（新建）

**Commit:** `61c0be5` + `605f562` + `72b0096`

**文件职责：** 向量去重引擎，提供纯函数 `deduplicate_entries()`，对一批 MemoryEntry 计算 pairwise cosine similarity，标记近重复项。

#### `deduplicate_entries()` (行 13-66)

```python
def deduplicate_entries(
    entries: List[MemoryEntry],
    embedding_model,
    threshold: float = 0.85,
    strategy: Literal["keep_longer", "keep_newer", "keep_both"] = "keep_longer"
) -> List[MemoryEntry]:
```

**核心逻辑（行 32-66）：**

| 行号 | 逻辑 |
|------|------|
| 32-33 | 快速路径：`not entries` 或 `strategy == "keep_both"` 时直接返回原列表 |
| 35-36 | 单条快速路径：直接返回 |
| 38-43 | **Embedding 计算**：调用 `embedding_model.encode_documents([e.lossless_restatement for e in entries])`。外层 `try/except` 捕获任何异常，打印警告并返回原列表（fail-open） |
| 46-50 | **向量归一化**：用 `np.linalg.norm` 计算每行范数，零范数位置设为 1.0 避免除零，然后归一化 |
| 53 | **相似度矩阵**：`np.dot(normalized, normalized.T)` 得到 NxN cosine similarity |
| 56-62 | **Pairwise 遍历**：只遍历上三角（`i < j`），当 `sim_matrix[i, j] > threshold` 时，调用 `_choose_winner()` 选择存活者，将 loser 的 `superseded_by` 设为 winner 的 `entry_id`（仅当 loser 当前未被标记时才写入） |
| 64-74 | **传递性链 canonicalization**（`72b0096` 补充）：构建 `entry_map = {e.entry_id: e}`，对每条已标记的 entry，沿着 `superseded_by` 链 walk 到 root winner，重写为直接指向 root。这避免了 A→B→C 的中间链问题 |
| 76-78 | **相似度计算 fail-open**：外层 `try/except` 捕获归一化/点乘/遍历中的任何异常，显式 `return entries` |

#### `_choose_winner()` (行 81-101)

```python
def _choose_winner(
    a: MemoryEntry,
    b: MemoryEntry,
    strategy: Literal["keep_longer", "keep_newer", "keep_both"]
) -> tuple[MemoryEntry, MemoryEntry]:
```

| 策略 | 规则 | 平手处理 |
|------|------|---------|
| `keep_longer` | `len(lossless_restatement)` 更大者胜 | 按 `entry_id` 字典序较小者胜 |
| `keep_newer` | 时间戳更晚者胜，`None` 时间戳必败 | 按 `entry_id` 字典序较小者胜 |

**类型注解修正（`72b0096`）：** `strategy` 参数类型从 `Literal["keep_longer", "keep_newer"]` 放宽为 `Literal["keep_longer", "keep_newer", "keep_both"]`，避免 mypy 警告（虽然 `keep_both` 在 `deduplicate_entries` 入口已过滤）。

---

### 2.3 `tests/test_vector_dedup.py`（新建）

**Commit:** `61c0be5` + `72b0096`

共 9 个测试：

| 测试名 | 验证点 |
|--------|--------|
| `test_empty_input` | `[]` → `[]` |
| `test_single_entry_unchanged` | 单条返回，无标记 |
| `test_merge_pair_above_threshold` | 相同 embedding → 较短者被标记为 superseded |
| `test_no_merge_below_threshold` | 正交 embedding → 均未被标记 |
| `test_keep_longer_strategy` | `keep_longer` 策略下短文本被长文本替代 |
| `test_keep_newer_strategy` | `keep_newer` 策略下旧时间戳被新时间戳替代 |
| `test_transitive_chain_canonicalized` | 3 条相同 embedding，最长者胜，其余两者均直接指向最长者 |
| `test_fail_open_on_embedding_error` | embedding 抛异常 → 原样返回 |
| `test_fail_open_on_similarity_error` | mock `np.linalg.norm` 抛异常 → 原样返回 |

---

### 2.4 `database/vector_store.py`

**Commit:** `5d715bc` + `8258b07`

#### Schema 变更

| 行号 | 改动 |
|------|------|
| 64-65 | `_init_table()` schema 中新增 `pa.field("superseded_by", pa.string())`，位于 `vector` 字段之后 |

#### `add_entries()` (行 142-143)

| 行号 | 改动 |
|------|------|
| 142-143 | 写入 LanceDB 的数据 dict 中增加 `"superseded_by": entry.superseded_by or ""`，将 Python `None` 映射为空字符串存储 |

#### `_results_to_entries()` (行 116)

| 行号 | 改动 |
|------|------|
| 116 | 解析结果时增加 `superseded_by=r.get("superseded_by") or None`。使用 `.get()` 兼容旧数据（无该字段时返回 `None`） |

#### `mark_superseded()` (行 248-261，新增方法)

```python
def mark_superseded(self, entry_ids: List[str], superseded_by: str):
```

| 行号 | 逻辑 |
|------|------|
| 249 | 空列表快速返回 |
| 253-258 | 逐条遍历 `entry_ids`，对单引号做 `replace("'", "''")` 转义，调用 `self.table.update(where=f"entry_id = '{safe_id}'", values={"superseded_by": superseded_by})` |
| 260-261 | 外层 `try/except` 捕获异常并打印错误 |

**注意：** LanceDB 的 `update` 目前只支持单条 where 条件，因此必须用循环而非批量更新。

#### 搜索方法过滤（所有 4 个搜索方法）

| 方法 | 行号 | 过滤方式 |
|------|------|---------|
| `semantic_search` | 163-165 | `.search(query_vector).where("superseded_by == ''", prefilter=True).limit(top_k)` |
| `keyword_search` | 183-185 | `.search(query).where("superseded_by == ''", prefilter=True).limit(top_k)` |
| `structured_search` | 226 | 在 `conditions` 列表末尾追加 `conditions.append("superseded_by == ''")`，再 `" AND ".join(conditions)`（`8258b07` 重构，避免字符串拼接的 fragility） |
| `get_all_entries` | 244-245 | 从 `.to_arrow().to_pylist()` 获取全部结果，在 Python 中过滤 `[r for r in results if r.get("superseded_by") == ""]`（`8258b07` 修正，避免 `.search().where().to_list()` 的潜在隐式 limit 问题） |

---

### 2.5 `tests/test_vector_store.py`

**Commit:** `8258b07`

在原有 7 个测试基础上新增 2 个测试：

| 测试名 | 验证点 |
|--------|--------|
| `test_mark_superseded_and_search_filtering` | 添加一条重复 entry → 用 `mark_superseded` 标记 → 验证 `get_all_entries` / `semantic_search` / `keyword_search` / `structured_search` 均不再返回被标记条目 |
| `test_results_to_entries_backward_compat` | 构造缺少 `superseded_by` 键的字典 → 验证 `_results_to_entries` 能正确解析且 `superseded_by` 为 `None` |

---

### 2.6 `cross/storage_lancedb.py`

**Commit:** `461e70d`

#### `_build_where_clause()` (行 304-324)

| 行号 | 改动 |
|------|------|
| 306 | `conditions = []` 改为 `conditions = ["superseded_by == ''"]`，所有通过该方法构建的 where 子句都会自动排除已 superseded 的条目 |
| 318-320 | 移除 `if not conditions: return None` 的提前返回，因为现在 conditions 至少有一个元素 |

**影响范围：** `_build_where_clause` 被 `semantic_search`、`keyword_search`、`get_entries_for_session`、`get_all_entries`、`count_entries` 共用，因此这 5 个方法都会自动过滤 superseded 条目。

#### `structured_search()` (行 512)

| 行号 | 改动 |
|------|------|
| 512 | 在构建完 `conditions` 列表后，追加 `conditions.append("superseded_by == ''")`，再 join 成 where_clause |

**注意：** `structured_search` 不调用 `_build_where_clause`（它自己组装条件），因此需要单独追加过滤条件。

---

### 2.7 `core/memory_builder.py`

**Commit:** `7b40dfd`

#### 新增 import (行 9, 12)

```python
from collections import defaultdict
from core.vector_dedup import deduplicate_entries
```

#### `process_window()` (行 148-165)

原逻辑（行 150-152）：
```python
if entries:
    self.vector_store.add_entries(entries)
    self.previous_entries = entries
    self.processed_count += len(window)
```

新逻辑：在 `add_entries` 前插入去重判断：

1. 检查 `config.ENABLE_DEDUP`（默认 `True`）
2. 若启用，调用 `deduplicate_entries(entries, self.vector_store.embedding_model, threshold=config.DEDUP_THRESHOLD, strategy=config.DEDUP_STRATEGY)`
3. 分离 `survivors`（`superseded_by is None`）和 `dropped`
4.  survivors → `add_entries()`
5.  dropped → 按 `superseded_by` 分组，调用 `vector_store.mark_superseded(dropped_ids, winner_id)`
6.  `self.previous_entries = entries`（注意：这里保留全部 entries，包括被标记的，作为下一轮 LLM 提示的上下文）

#### `process_remaining()` (行 182-200)

与 `process_window()` 完全相同的去重逻辑模式，作用于 `self.dialogue_buffer` 的剩余条目。

#### `_process_windows_parallel()` (行 398-434)

原逻辑（行 364-371）：所有 worker 返回的 entries 汇总到 `all_entries` 后，直接 `add_entries(all_entries)`。

新逻辑：在统一 batch 写入前，执行与串行路径完全相同的去重流程：
1. `deduplicate_entries(all_entries, ...)`
2. 分离 survivors / dropped
3. survivors → `add_entries()`
4. dropped → `mark_superseded()`
5. 打印 `[Parallel Processing] Marked {len(dropped)} duplicates as superseded`
6. `self.previous_entries = all_entries[-10:]`（保留最后 10 条作为上下文）

**关键不变量：** Worker 线程本身不做任何修改，去重完全发生在主线程的统一 batch 阶段，因此不存在并发竞争。

---

### 2.8 `cross/session_manager.py`

**Commit:** `432cbb0`

#### 新增 import (行 23)

```python
from core.vector_dedup import deduplicate_entries
```

#### `_run_simplemem_pipeline()` (行 694-736)

原逻辑（行 696-709）：SimpleMem finalize 产出的 `memory_entries` 直接通过 `self._vector_store.add_entries()` 写入 Cross LanceDB。

新逻辑：在写入前插入去重：

1. 内联 `import config as _config`
2. 检查 `_config.ENABLE_CROSS_DEDUP`（默认 `True`）
3. 若启用，调用 `deduplicate_entries(memory_entries, self._vector_store.embedding_model, threshold=_config.CROSS_DEDUP_THRESHOLD, strategy=_config.DEDUP_STRATEGY)`
4. 分离 survivors / dropped
5. survivors → `add_entries()`（参数不变）
6. dropped → 按 winner 分组，逐条调用 `self._vector_store.mark_superseded(d_id, winner_id)`
7. 日志：`logger.info("Marked %d cross-session entries as superseded", len(dropped))`
8. 返回 `len(survivors)` 而非 `len(memory_entries)`

**注意：** Cross 的 `mark_superseded` 签名是 `(old_entry_id: str, new_entry_id: str)`，只接受单条 ID，因此必须用循环处理。而 core VectorStore 的 `mark_superseded` 是 `(entry_ids: List[str], superseded_by: str)`，支持批量。

---

### 2.9 `config.py.example`

**Commit:** `13c38e0`

在 `Planning and Reflection Configuration` 与 `LLM-as-Judge Configuration` 之间插入新的配置块：

```python
# ============================================================================
# Deduplication Configuration
# ============================================================================

# Enable write-time deduplication for core memory pipeline
ENABLE_DEDUP = True

# Cosine similarity threshold (0.0 - 1.0).
# Higher = stricter, fewer duplicates caught.
# Lower = looser, may merge non-duplicates.
DEDUP_THRESHOLD = 0.85

# Merge strategy: "keep_longer" | "keep_newer" | "keep_both"
# keep_longer  - retain the entry with more detailed restatement
# keep_newer   - retain the entry with the more recent timestamp
# keep_both    - disable deduplication for this path
DEDUP_STRATEGY = "keep_longer"

# Enable write-time deduplication for cross-session memory
ENABLE_CROSS_DEDUP = True

# Cross-session uses a lower threshold because observations are more fragmented
CROSS_DEDUP_THRESHOLD = 0.80
```

**注意：** `config.py` 也被同步修改（同内容），但由于 `.gitignore` 中排除了 `config.py`，只有 `config.py.example` 进入版本控制。

---

## 三、测试覆盖

### 3.1 单元测试

- `tests/test_vector_dedup.py`：**9 个测试全部通过**
  - 覆盖空输入、单条、合并、阈值边界、keep_longer、keep_newer、传递性链、embedding 失败、similarity 失败

### 3.2 集成测试

- `tests/test_vector_store.py`（脚本运行）：**9 个测试全部通过**
  - 包含新增的 `test_mark_superseded_and_search_filtering` 和 `test_results_to_entries_backward_compat`
- `cross/tests/`：**127 个测试全部通过**

### 3.3 配置验证

- `config.ENABLE_DEDUP`、`DEDUP_THRESHOLD`、`DEDUP_STRATEGY`、`ENABLE_CROSS_DEDUP`、`CROSS_DEDUP_THRESHOLD` 均可正常读取

---

## 四、与现有系统的关系

| 系统 | 关系 |
|------|------|
| `cross/consolidation.py` | **并存，不冲突**。Inline dedup 在每次写入时以 0.85 阈值剔除明显重复；consolidation 作为后台任务以 0.95 阈值做全局深度清理。inline dedup 减少了 consolidation 的工作量 |
| `core/hybrid_retriever.py` | **无改动**。检索侧去重不在本次范围。VectorStore 的搜索方法已过滤 superseded 条目，检索结果自然不含重复 |
| `cross/storage_sqlite.py` | **无改动**。SQLite 中的 observations 和 summaries 不在去重范围内 |
| 现有 LanceDB 表 | **向后兼容**。`_results_to_entries` 使用 `.get("superseded_by")` 处理缺失字段；新表有该字段，旧数据无该字段时解析为 `None` |

---

## 五、已知限制

1. **LanceDB `update` 单条限制**：`mark_superseded` 目前逐条调用 `table.update()`，N 条即 N 次 round-trip。LanceDB 若未来支持批量 `update`/`merge_insert` 可优化
2. **Cross `mark_superseded` 单条签名**：CrossSessionVectorStore 的 `mark_superseded(old_entry_id, new_entry_id)` 只接受单条，session_manager 中需循环调用
3. **阈值选择**：0.85（core）/ 0.80（cross）是基于启发式的默认值，实际效果取决于 embedding 模型。不同模型可能需要微调
4. **内容不合并**：去重仅标记替代关系，不合成内容。内容合并留给 consolidation 层
