from fastapi.testclient import TestClient
from main import app

client=TestClient(app)

def ensure_user_registered(username="test_user_123",password="123456"):
    client.post("/register",json={"username":username,"password":password})

def test_health():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"

def test_register_and_login():
    ensure_user_registered()
    r=client.post("/login",json={"username":"test_user_123","password":"123456"})
    assert r.status_code==200
    assert "access_token" in r.json()

def test_ask_requires_auth():
    #未带Token访问/ask应401
    r=client.post("/ask",json={"question":"什么是自注意力？"})
    assert r.status_code==401

def test_ask_with_token():
    ensure_user_registered()
    r=client.post("/login",json={"username":"test_user_123","password":"123456"})
    token=r.json()["access_token"]
    headers={"Authorization":f"Bearer {token}"}
    r=client.post("/ask",json={"question":"Transformer为什么用自注意力？"},headers=headers)
    assert r.status_code==200
    data=r.json()
    assert "answer" in data
    assert "sources" in data