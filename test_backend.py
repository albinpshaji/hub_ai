import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post("http://127.0.0.1:8004/api/v1/chat/stream", json={
                "messages": [{"role": "user", "content": "who is abel?"}],
                "user_id": "string",
                "use_rag": False
            })
            with open("test_out.txt", "w") as f:
                f.write(str(resp.status_code) + "\n")
                async for chunk in resp.aiter_text():
                    f.write(chunk)
        except Exception as e:
            with open("test_out.txt", "w") as f:
                f.write("ERROR: " + str(e))

asyncio.run(main())
