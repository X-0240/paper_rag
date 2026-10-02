# 论文知识库问答系统（RAG）

把 10 篇 LLM 核心论文做成一个可问答的知识库：中文提问 → 多语言向量检索英文论文切片 → DeepSeek 生成带引用来源的回答。
接口用 JWT 保护，相同问题走缓存，模型失败有指数退避重试与兜底。

**主指标**：论文级 HitRate@5 = **86%**（Top-1 = 69%）；100 道中文测试题，题库与参考答案由我构造并逐题标注来源。

## 这个项目解决什么问题

- **中文提问要能命中英文论文**：用多语言 Embedding 做跨语言匹配，不需要先把问题翻译成英文。
- **回答要能溯源**：切片时记录「论文 + 章节」，检索结果与回答都带来源，而不是只给一段没有出处的文字。
- **重复提问不该重复烧钱**：相同问题复用上次结果，省一次向量编码与一次模型调用。
- **这是第一个项目**：链路最短、最容易讲清；后续的手写 ReAct Agent 与 300 篇语料版本是在它之上演进的。

## 这个仓库能跑什么（克隆前必读）

**本仓库只含代码与配置模板。以下内容不在仓库里，所以克隆后无法直接跑通：**

| 不在仓库里 | 原因 | 怎么补 |
|---|---|---|
| 10 篇论文 PDF | 论文原文有版权，体积也大 | 自行下载对应论文，放进 `.env` 的 `PAPERS_DIR` |
| FAISS 索引与元数据（`*.faiss` / `*.json`） | 体积大，可由脚本重建 | 跑 `build_papers_faiss.py` 重新生成 |
| 多语言 Embedding 模型权重 | 体积大 | 自行下载（如 `paraphrase-multilingual-MiniLM-L12-v2`），路径写进 `.env` 的 `MODEL_PATH` |
| `.env` | 含 API Key 与本机路径 | 按下面「快速开始」自己写一份 |

**克隆后能直接确认的**：`python -m pytest test_api.py -q` 里除 `test_ask_with_token` 之外的 3 条接口用例（健康检查、注册登录、未带 Token 返回 401），它们不需要模型额度。

## 快速开始

```bash
pip install -r requirements.txt
```

`.env` 需要这些键（`SECRET_KEY` 与 `DEEPSEEK_API_KEY` 填自己的）：

```ini
SECRET_KEY=<JWT 签名密钥>
ALGORITHM=HS256
DEEPSEEK_API_KEY=<自己的 DeepSeek Key>
PAPERS_DIR=<10 篇论文 PDF 所在目录>
MODEL_PATH=<多语言 Embedding 模型目录>
FAISS_PATH=<索引输出前缀，不带扩展名，路径必须全英文>
TEST_QUESTIONS_FILE=<100 题题库文件>
TEST_REFERENCES_FILE=<参考答案文件>
```

```bash
python build_papers_faiss.py       # 建库：解析 PDF → 按章节切片 → 向量 → FAISS 索引
python evaluate_papers_faiss.py    # 用 100 道中文题评估检索命中率
python main.py                     # 起服务：http://127.0.0.1:8000
python -m pytest test_api.py -q    # 接口测试
```

`rag_papers.py` 是命令行问答入口，不想起服务时可以用它验证检索与生成。

两个坑写在这里：**FAISS 不支持中文路径**，所以 `FAISS_PATH` 必须是纯英文目录；**索引维度必须与 `MODEL_PATH` 的模型一致**，换了模型要重建索引。

## 架构

```
10 篇论文 PDF
  │ PyMuPDF 解析：按 blocks 识别章节标题，正文按标题切分
  ▼
切片（500 字符 + 100 重叠，每条记录论文名与章节名）
  │ 多语言 Embedding 模型编码，向量归一化
  ▼
FAISS IndexFlatIP（归一化后内积 = 余弦相似度）
  │
中文问题 → 编码为向量 → Top-5 切片（带论文 + 章节 + 原文）
  │ 拼成上下文
  ▼
DeepSeek 生成回答 + 引用来源
```

模块职责：

| 模块 | 职责 |
|---|---|
| `build_papers_faiss.py` | 建库：PDF 解析、章节切分、向量化、写 FAISS 索引与元数据 |
| `rag_papers.py` | 检索与生成的命令行实现 |
| `evaluate_papers_faiss.py` | 100 道中文题的检索评估（论文级 / 章节级 HitRate） |
| `main.py` | FastAPI 服务：JWT 认证、问答接口、缓存与重试 |
| `test_api.py` | 接口测试（健康检查、注册登录、鉴权、带 Token 问答） |

## 目录结构

```
paper_rag/
├── build_papers_faiss.py        # 建库脚本
├── evaluate_papers_faiss.py     # 检索评估
├── rag_papers.py                # 命令行问答
├── main.py                      # FastAPI 服务
├── test_api.py                  # 接口测试
├── requirements.txt
└── README.md  LICENSE

不进仓库（由 .gitignore 挡住）：.env（含密钥）、chroma_data/ 与 papers_chroma_data/（早期 Chroma 方案的残留）、
__pycache__/、.pytest_cache/、_archive/（学习过程文件），以及论文 PDF、索引与模型权重
```

## 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/register` | POST | 注册（bcrypt 加密密码） |
| `/login` | POST | 登录，返回 JWT Token |
| `/ask` | POST | 论文问答，需要 `Authorization: Bearer <token>` |
| `/health` | GET | 健康检查 |

## 评估结果

| 指标 | 结果 | 口径 |
|---|---|---|
| 论文级 HitRate@5 | **86%**（Top-1 69%） | 100 道中文题，题库由我构造，参考答案逐题标注来源 |
| 章节级 HitRate@5 | 13% | 同上；低的原因是问题粒度与章节粒度错配，见「已知边界」 |

数字是本机实测值，随模型、语料与题目版本变化，不承诺复现。

## 已知边界

- **章节级命中率低**：问题问的是"某个方法的原理"，切片却是整节文本，粒度对不上——论文级 86%、章节级只有 13%，计划用 Rerank 精排解决。
- **公式与表格会丢**：PyMuPDF 只能拿文本层，公式、表格、图注内容取不到。
- **认证状态存在进程内存**：注册用户重启即失效，生产要换成数据库。
- 引用来源偶有无关论文混入，是检索粒度的另一个表现。

## 协作与反馈

个人项目，没有开放的贡献流程；发现问题或有建议请开 [Issue](https://github.com/X-0240/paper_rag/issues)。

## 许可与第三方素材

本仓库代码与文档采用 [MIT 许可](LICENSE)。以下内容**不在仓库内**，也不在本许可的授权范围内：

- **论文原文与 PDF**：版权归原作者与出版方，需自行按需下载。
- **模型权重**：多语言 Embedding 模型由各自原仓库发布并遵循其自身许可。
