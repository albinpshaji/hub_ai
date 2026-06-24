import httpx
try:
    resp = httpx.post("http://127.0.0.1:11434/api/chat", json={"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hello"}]}, timeout=120)
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print(e)
