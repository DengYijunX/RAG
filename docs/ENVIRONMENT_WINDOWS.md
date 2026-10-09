# Windows 环境说明与已知限制

记录日期：2026-10-09（Asia/Shanghai）。本文只描述本机环境的修复与限制，不含任何凭据内容。

## 修复记录

### 1. 项目虚拟环境 `.venv` 的基座被删除

`.venv/pyvenv.cfg` 原先指向 `C:\software\bin\Anaconda`，该目录已不存在，导致 `.venv\Scripts\python.exe` 无法启动（`No Python at ...`）。已把基座改为本机同 ABI 的 CPython 3.11.15（`%APPDATA%\uv\python\cpython-3.11.15-windows-x86_64-none`），venv 内 230 个既有依赖包全部保留、无需重装。

`.venv/` 已在 `.gitignore` 中，此修复不进入仓库。换机器或重装依赖按 `docs/CLOUD_STARTUP.md` 重建即可。

### 2. Git 远程地址不可用

仓库 `origin` 原为 `git@github-deng:DengYijunX/RAG.git`，但 SSH 配置里没有 `github-deng` 这个 Host 别名，且本机 22 端口被网络策略拦截（`Connection closed by 198.18.0.5 port 22`）。已改为 HTTPS：

```powershell
git remote set-url origin https://github.com/DengYijunX/RAG.git
```

如需改回 SSH，请先在 `%USERPROFILE%\.ssh\config` 中补一个 `Host github-deng` 别名。

### 3. 沙箱下 pytest 无法建临时目录

DSH 沙箱不允许向 `C:\WINDOWS\TEMP\dsh-*` 写入，且对以 `0o700` 模式创建的目录拒绝后续写入（`tempfile.mkdtemp` 正是这种模式），表现为 `PermissionError` 与 `sqlite3.OperationalError: unable to open database file`。

`tools/testenv/sitecustomize.py` 通过 `PYTHONPATH` 注入两个兼容处理：把临时目录指向工作区内 `.tmp/`，并把目录创建模式改为宽松模式。它只用于本机跑测试，不是应用代码：

```powershell
$env:PYTHONPATH="C:\Users\AD钙\Desktop\Projects\RAG\tools\testenv"
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=".tmp\all"
```

同时 `pytest` 需要显式用 `--basetemp` 指向工作区内目录；不要使用 `--basetemp` 指向工作区外。

### 4. 测试依赖 `pytest` 缺失

`.venv` 中没有 pytest，已通过 `uv pip install --python .venv\Scripts\python.exe pytest` 安装（`uv` 的缓存目录需指到工作区内，否则报 `拒绝访问`）。`python-multipart` 已加入 `requirements.txt` 与 `pyproject.toml`，是文件上传接口的必要依赖。

## 已知限制：原生向量索引在本机批量写入时崩溃

ChromaDB 依赖的原生扩展 hnswlib 在本机**写入较大批量向量时崩溃**，`0xC0000005` 访问违例。1024 维下的实测边界：单次 `add` 2 / 64 / 96 条成功（`add` + `search` 均正常），128 / 200 / 749 条崩溃。也就是说小批量可以写入，本项目会计语料的 749 片段索引无法在本机建立。

已排除的可能：

| 变量 | 尝试 | 结果 |
| --- | --- | --- |
| Python 版本 | 3.11.15（项目 venv 与全新 venv）、3.12.2 | 大批量均崩溃 |
| numpy | 1.26.4 (`numpy<2`)、2.5.3 | 大批量均崩溃 |
| hnswlib 版本 | 0.8.0（有 wheel）；0.7.0 / 0.6.2 无可用 wheel，源码编译失败 | 大批量崩溃 |
| 线程数 | `OMP_NUM_THREADS=1`、`HNSWLIB_NUM_THREADS=1` | 大批量均崩溃 |
| 数据规模 | 128 / 200 / 749 条（1024 维） | 均崩溃；≤96 条正常 |
| 沙箱权限 | 工作区沙箱与一次性完全权限 | 均崩溃 |

结论：这是**机器级**问题，与本项目代码无关，也不是沙箱造成。因此本机无法运行 `tests/test_knowledge.py`（需要构建 749 个片段的 Chroma 索引）与使用真实向量库的端到端索引验收；`tests/test_store.py` 只用 2 条 3 维向量，可以正常运行。

在能正常运行的环境（Docker / Linux 主机 / 另一台机器）执行：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest experiments/rag-baseline/evaluation -q
RAG_ACCEPTANCE_REAL=1 .venv/bin/python experiments/rag-baseline/evaluation/acceptance_third_corpus.py
```

## 清理提示

以下目录已在 `.gitignore` 中，均为本机临时产物：`.tmp/`、`.uv-cache/`、`.pytest_cache/`、`.dsh-acl-recovery/`。

2026-10-09 已清理 `.tmp/`（304 MB）、`.uv-cache/`（324 MB）、`.pytest_cache/`、`.dsh-acl-recovery/`。**残留需管理员权限删除**：`.tmp/` 下沙箱创建的 `pytest-of-AD钙` 等 6 个空目录，以及 `.uv-cache/` 剩余的 `builds-v0` / `interpreter-v4` / `sdists-v9`（合计约 96.7 MB）——当前进程对它们没有删除权限，`Remove-Item` 报 `Access is denied`。它们都不在版本控制内，不影响仓库与测试。
