import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("http://127.0.0.1:11434/api/chat", json={"model": "llama3.2:3b", "messages": []})
            print(resp.status_code)
            print(resp.text)
        except Exception as e:
            print(e)

asyncio.run(main())
