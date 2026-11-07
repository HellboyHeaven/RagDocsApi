async def test_health(client):
    r = await client.get("/health")
    assert r.json() == {"status": "ok"}