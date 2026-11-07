import pytest

from tests.conftest import make_hit


async def test_search_returns_hits(client, fake_embed, monkeypatch):
    async def fake_search(session, vector, k):
        assert k == 2  # the k query parameter reaches the CRUD layer
        return [make_hit(1, 0.1), make_hit(2, 0.3)]

    monkeypatch.setattr("app.routers.search.crud.search_chunks", fake_search)

    r = await client.get("/search", params={"q": "something", "k": 2})

    assert r.status_code == 200
    assert [h["distance"] for h in r.json()] == [0.1, 0.3]


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 21}])
async def test_search_validates_params(client, params):
    r = await client.get("/search", params=params)
    assert r.status_code == 422