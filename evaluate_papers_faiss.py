import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

#加载索引+元数据
base=r"C:\Users\Public\papers_faiss"
index=faiss.read_index(base+".faiss")
meta=json.load(open(base+".json",encoding="utf-8"))
sources=meta["sources"]
sections=meta["sections"]
model=SentenceTransformer(r"D:\Users\徐海生\Documents\agent项目\multi-minilm-model")

#读取100题（问题+原始引用）
q_file=r"C:\Users\徐海生\.codex\attachments\fecb0a01-84bc-4dfc-a920-50da471dd825\pasted-text.txt"
content=open(q_file,encoding="utf-8").read()
questions=re.findall(r"问题：(.+)",content)
orig_refs=re.findall(r"引用来源：(.+)",content)

#读取修正引用
v_file=r"C:\Users\徐海生\.codex\attachments\a9eb6673-f6d6-40a1-a5d5-ea125acd5d86\pasted-text.txt"
v_content=open(v_file,encoding="utf-8").read()
fixed_refs=[]
for block in re.split(r"(?=第\d+题)",v_content):
    if not block.strip():
        continue
    m=re.search(r"修正后引用：(.*)",block)
    if m and m.group(1).strip():
        fixed_refs.append(m.group(1).strip())
    else:
        m2=re.search(r"原始引用：(.*)",block)
        if m2:
            fixed_refs.append(m2.group(1).strip())
print(f"题目:{len(questions)}, 引用:{len(fixed_refs)}")

#论文映射
paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need",
    "BERT":"BERT","Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention",
    "GraphRAG":"GraphRAG","LoRA":"LoRA","RAG Original":"RAG_Original_Paper",
    "ReAct":"ReAct","GPT2":"GPT2","GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}

#章节匹配
def section_match(section,expected_section):
    def norm(s):
        s=re.sub(r"^[\d.]+\s*","",s.lower())
        s=re.sub(r"^(figure|table|abstract|appendix|proposition|theorem|section)\s*[\d.]*\s*","",s)
        return re.sub(r"[^a-z ]","",s).strip()
    e=norm(expected_section)
    a=norm(section)
    if not e or not a:
        return False
    return e in a or a in e

#评估
hits_1=0; hits_3=0; hits_5=0
p_hits_1=0; p_hits_3=0; p_hits_5=0
details=[]
for i,(q,ref) in enumerate(zip(questions,fixed_refs)):
    parts=re.split(r"\s+-\s+",ref,maxsplit=1)
    if len(parts)<2:
        details.append((i+1,q,"","","格式异常",""))
        continue
    paper_part,section_part=parts
    source=None
    for key,src in paper_map.items():
        if key.lower() in paper_part.lower():
            source=src; break
    if not source:
        details.append((i+1,q,paper_part,section_part,"未知论文",""))
        continue

    #检索
    q_vec=model.encode([q])
    q_vec=q_vec/np.linalg.norm(q_vec)
    scores,idx=index.search(q_vec.astype("float32"),5)
    res_sources=[sources[j] for j in idx[0]]
    res_sections=[sections[j] for j in idx[0]]

    hit1=any(s==source and section_match(sec,section_part) for s,sec in zip(res_sources[:1],res_sections[:1]))
    hit3=any(s==source and section_match(sec,section_part) for s,sec in zip(res_sources[:3],res_sections[:3]))
    hit5=any(s==source and section_match(sec,section_part) for s,sec in zip(res_sources[:5],res_sections[:5]))
    hits_1+=hit1; hits_3+=hit3; hits_5+=hit5

    #论文级统计（只看论文，不看章节）
    p_hit1=any(s==source for s in res_sources[:1])
    p_hit3=any(s==source for s in res_sources[:3])
    p_hit5=any(s==source for s in res_sources[:5])
    p_hits_1+=p_hit1; p_hits_3+=p_hit3; p_hits_5+=p_hit5

    if not hit5:
        details.append((i+1,q,source,section_part,"未命中",res_sections[:3]))

total=len(questions)
print(f"\n章节级 Hit Rate@1={hits_1/total:.2%} @3={hits_3/total:.2%} @5={hits_5/total:.2%}")
print(f"\n=== 未命中Top5({len(details)}条) ===")
print(f"论文级 Hit Rate@1={p_hits_1/total:.2%} @3={p_hits_3/total:.2%} @5={p_hits_5/total:.2%}")
for i,q,src,sec,st,secs in details[:25]:
    print(f"第{i}题 {q[:40]}")
    print(f"  预期:{sec}")
    print(f"  实际返回:{secs}")
