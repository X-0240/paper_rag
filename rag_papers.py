import logging
import os
import json
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#FAISS索引+元数据（英文路径）
FAISS_PATH=r"C:\Users\Public\papers_faiss"
index=faiss.read_index(FAISS_PATH+".faiss")
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
sources=meta["sources"]
sections=meta["sections"]
documents=meta["documents"]

#多语言模型（中文问题检索英文论文）
model=SentenceTransformer(r"D:\Users\徐海生\Documents\agent项目\multi-minilm-model")

#DeepSeek配置
api_key=os.getenv("DEEPSEEK_API_KEY")
headers={
    "Authorization":f"Bearer {api_key}",
    "Content-Type":"application/json"
}

#检索：中文问题→向量→FAISS搜索→返回切片+来源
def search_papers(query,k=5):
    q_vec=model.encode([query])
    q_vec=q_vec/np.linalg.norm(q_vec)
    scores,idx=index.search(q_vec.astype("float32"),k)
    results=[]
    for score,j in zip(scores[0],idx[0]):
        results.append({
            "document":documents[j],
            "source":sources[j],
            "section":sections[j],
            "score":float(score)
        })
    return results

#调用DeepSeek
def call_deepseek(messages):
    response=requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json={"model":"deepseek-v4-flash","messages":messages,"temperature":0.7},
        timeout=30
    )
    if not response.ok:
        logger.error(f"请求失败，状态码：{response.status_code}")
        logger.error(f"错误响应：{response.text}")
    response.raise_for_status()
    return response.json()

#重试
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=1,max=10),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def safe_call_deepseek(messages):
    return call_deepseek(messages)

#RAG问答：检索→拼上下文→生成
def rag_query(question):
    results=search_papers(question,k=5)
    #拼上下文：论文+章节+切片内容
    context="\n\n".join(
        f"[来源:{r['source']} - {r['section']}]\n{r['document']}"
        for r in results
    )
    messages=[
        {"role":"system","content":"你是一个论文知识问答助手。根据提供的论文内容回答用户问题，回答要引用来源。如果信息不足，明确说'根据现有论文无法回答'。"},
        {"role":"user","content":f"以下是相关的论文内容：\n{context}\n\n用户问题：{question}"}
    ]
    result=safe_call_deepseek(messages)
    answer=result["choices"][0]["message"]["content"]
    tokens=result["usage"]["total_tokens"]
    #返回答案+引用来源
    refs=[f"{r['source']} - {r['section']}" for r in results if r["score"]>0.3]
    logger.info(f"问题:{question}, token:{tokens}")
    return answer,refs,tokens

#主流程：命令行问答
if __name__=="__main__":
    print("论文知识问答助手，输入问题（Ctrl+C退出）")
    while True:
        question=input("问题：")
        if not question:
            continue
        try:
            answer,refs,tokens=rag_query(question)
            print(f"\n回答：{answer}")
            if refs:
                print(f"\n参考来源：{'; '.join(refs)}")
            print(f"Token：{tokens}\n")
        except Exception as e:
            logger.error(f"问答失败：{e}")
            print("服务暂时不可用，请稍后再试")