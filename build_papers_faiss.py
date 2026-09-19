import os
import re
import json
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
#论文PDF目录
papers_dir=os.getenv("PAPERS_DIR")
#多语言模型
model=SentenceTransformer(os.getenv("MODEL_PATH"))
#识别章节标题
def is_heading(block):
    t=block.strip().replace("\n"," ")
    if len(t)<5 or len(t)>80:
        return False
    if not re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]",t):
        return False
    if re.search(r"[\U0001D400-\U0001D7FF]",t):
        return False
    if re.match(r"^\d+\.?\s+(compute|set|let|for|while|if|return)\b",t,re.I):
        return False
    return True

#从PDF按章节提取
def extract_sections(pdf_path):
    doc=fitz.open(pdf_path)
    sections=[]
    current_title="Abstract"
    current_text=""
    for page in doc:
        for block in page.get_text("blocks"):
            text=block[4].strip()
            if not text:
                continue
            if is_heading(text):
                if current_text.strip():
                    sections.append((current_title,current_text.strip()))
                current_title=text.replace("\n"," ")
                current_text=""
            else:
                current_text+=text+"\n"
    if current_text.strip():
        sections.append((current_title,current_text.strip()))
    doc.close()
    return sections

#切片（固定长度+重叠；语义切实验曾致论文级HitRate@5从93%降到92%，已回退）
def chunk_section(text,chunk_size=500,overlap=100):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunks.append(text[start:end])
        start+=chunk_size-overlap
    return chunks

#构建
all_chunks=[]
sources=[]
sections=[]
for fname in sorted(os.listdir(papers_dir)):
    if not fname.endswith(".pdf"):
        continue
    source=fname.replace(".pdf","")
    sec_list=extract_sections(os.path.join(papers_dir,fname))
    for title,sec_text in sec_list:
        if len(sec_text)<50:
            continue
        chunks=chunk_section(sec_text)
        for j,chunk in enumerate(chunks):
            all_chunks.append(chunk)
            sources.append(source)
            sections.append(title)
    print(f"{source}: {len(sec_list)}个章节")
print(f"切片总数:{len(all_chunks)}")

#转向量+归一化（归一化后内积=余弦相似度）
embeddings=model.encode(all_chunks,show_progress_bar=True)
embeddings=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)

#构建FAISS索引
dim=embeddings.shape[1]
index=faiss.IndexFlatIP(dim)
index.add(embeddings.astype("float32"))

#持久化：索引+元数据
base=os.getenv("FAISS_PATH")
faiss.write_index(index,base+".faiss")
with open(base+".json","w",encoding="utf-8") as f:
    json.dump({"sources":sources,"sections":sections,"documents":all_chunks},f,ensure_ascii=False)
print(f"已保存{len(all_chunks)}条到{base}")
