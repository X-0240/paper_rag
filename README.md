# 论文知识库问答系统（RAG）

基于 10 篇 LLM 核心论文构建的检索增强问答服务：中文提问，英文论文检索，DeepSeek 生成回答并附带引用来源。

## 功能特性

- 中文问题检索英文论文（多语言 Embedding 模型跨语言匹配）
- 按论文章节切分，检索结果带论文和章节来源
- JWT 认证保护问答接口
- 相同问题结果缓存（省 Token）
- 指数退避重试 + 异常兜底
- 完整日志（问题、Token 消耗、失败原因）

## 技术栈

- Python 3.10 / FastAPI
- FAISS 向量检索（IndexFlatIP，归一化后=余弦相似度）
- sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2，多语言）
- PyMuPDF（PDF 解析：正文提取 + 章节结构识别）
- DeepSeek API
- JWT / bcrypt 认证
- pytest

## 架构流程

```
10 篇论文 PDF
  ↓ PyMuPDF 按章节提取（blocks 模式识别章节标题）
切片（500字符+100重叠，记录论文+章节）
  ↓ 多语言模型转向量
FAISS 索引
  ↓ 检索
中文问题 → 向量 → FAISS Top-5 → 论文+章节+切片
  ↓ 生成
拼上下文 → DeepSeek → 回答 + 引用来源
```

## 评估结果

- 100 道中文测试题（基于 10 篇论文生成，引用来源经验证）
- 论文级 Hit Rate@5 = 86%（Top-1 = 69%）
- 章节级 Hit Rate@5 = 13%（已知边界：问题粒度与章节粒度错配）

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/register` | POST | 注册（bcrypt 加密密码） |
| `/login` | POST | 登录，返回 JWT Token |
| `/ask` | POST | 论文问答（需 Bearer Token） |
| `/health` | GET | 健康检查 |

## 快速开始

```bash
# 1. 建论文知识库（FAISS 索引 + 元数据）
python build_papers_faiss.py

# 2. 评估检索
python evaluate_papers_faiss.py

# 3. 启动 API 服务
python main.py

# 4. 运行测试
python -m pytest test_api.py -v
```

## 项目结构

```
build_papers_faiss.py     建库（解析→切片→向量→FAISS）
evaluate_papers_faiss.py  100题评估
rag_papers.py             命令行问答
main.py                   FastAPI 服务（JWT + 问答接口）
test_api.py               接口测试
papers_faiss.faiss        FAISS 索引（英文路径，FAISS 不支持中文路径）
papers_faiss.json         元数据（论文/章节/原文）
multi-minilm-model/       多语言 Embedding 模型
```

## 已知边界与下一步

- 引用来源偶有无关论文混入（检索粒度问题，计划用 Rerank 精排）
- 章节级命中率低（问题粒度与章节粒度错配）
- 公式/表格部分丢失（PyMuPDF 文本层局限）
- 认证用内存存储，生产应换数据库
- 后续：手写 ReAct 多 Agent，将 RAG 封装为 Agent 可调用工具
