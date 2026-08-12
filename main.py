import json
import logging
import os
from datetime import datetime, timedelta

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

app=FastAPI()

#CORS（前端可调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#JWT配置
SECRET_KEY=os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("请在.env里配置SECRET_KEY")
ALGORITHM="HS256"
pwd_context=CryptContext(schemes=["bcrypt"],deprecated="auto")
security=HTTPBearer(auto_error=False)

#临时用户存储（内存，生产应换数据库）
fake_users_db={}

#加载FAISS索引+元数据
FAISS_PATH=os.getenv("FAISS_PATH")
index=faiss.read_index(FAISS_PATH+".faiss")
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
sources=meta["sources"]
sections=meta["sections"]
documents=meta["documents"]

#多语言模型（启动时加载一次，常驻内存）
model=SentenceTransformer(os.getenv("MODEL_PATH"))
#DeepSeek配置
api_key=os.getenv("DEEPSEEK_API_KEY")
headers={
    "Authorization":f"Bearer {api_key}",
    "Content-Type":"application/json"
}

#缓存：相同问题不重复调API
cache={}

#Pydantic模型
class UserRegister(BaseModel):
    username:str
    password:str

class Question(BaseModel):
    question:str

#检索：中文问题→向量→FAISS搜索
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

#RAG问答
def rag_answer(question):
    results=search_papers(question,k=5)
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
    refs=[f"{r['source']} - {r['section']}" for r in results if r["score"]>0.3]
    logger.info(f"问题:{question}, token:{tokens}")
    return answer,refs,tokens

#JWT验证
def verify_token(credentials: HTTPAuthorizationCredentials|None=Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401,detail="未提供Token")
    token=credentials.credentials
    try:
        payload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401,detail="无效的Token或已过期")






#注册
@app.post("/register")
def register(user:UserRegister):
    if user.username in fake_users_db:
        raise HTTPException(status_code=409,detail="用户名已存在")
    hashed=pwd_context.hash(user.password)
    fake_users_db[user.username]={"username":user.username,"password":hashed}
    logger.info("用户%s注册成功",user.username)
    return {"message":f"用户{user.username}注册成功"}

#登录
@app.post("/login")
def login(user:UserRegister):
    username=user.username
    password=user.password
    if username not in fake_users_db:
        raise HTTPException(status_code=401,detail="用户不存在，请先注册")
    if not pwd_context.verify(password,fake_users_db[username]["password"]):
        raise HTTPException(status_code=401,detail="密码错误")
    exp=datetime.utcnow()+timedelta(hours=1)
    token=jwt.encode({"sub":username,"exp":exp},SECRET_KEY,algorithm=ALGORITHM)
    logger.info("用户%s登录成功",username)
    return {"access_token":token,"token_type":"bearer"}

#问答接口（需认证）
@app.post("/ask")
def ask(question:Question,username:str=Depends(verify_token)):
    q=question.question
    if q in cache:
        logger.info("命中缓存:%s",q)
        return cache[q]
    try:
        answer,refs,tokens=rag_answer(q)
        result={"question":q,"answer":answer,"sources":refs,"tokens":tokens}
        cache[q]=result
        return result
    except Exception as e:
        logger.error(f"问答失败:{e}")
        raise HTTPException(status_code=503,detail="服务暂时不可用，请稍后再试")

#健康检查
@app.get("/health")
def health():
    return {"status":"ok","collection_size":len(documents)}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)