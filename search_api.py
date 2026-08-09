"""
search_api.py - lokalne API nad Chroma plus prosty interfejs webowy.

Start:
    python -m uvicorn search_api:app --host 127.0.0.1 --port 8000

Potem otworz w przegladarce: http://127.0.0.1:8000

Endpointy:
    GET  /                  -> interfejs webowy (ui.html)
    GET  /health            -> stan Ollama, liczba fragmentow, aktualne parametry
    POST /search            -> sam retrieval (uzywa tego n8n oraz tryb "tylko wyszukiwanie")
    POST /ask               -> pelny RAG: retrieval + generowanie

Nasluchuje TYLKO na 127.0.0.1 - nic nie wychodzi poza ta maszyne.
Interfejs jest serwowany z tego samego procesu, wiec nie ma problemu z CORS.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import core
import pamiec

app = FastAPI(title="Personal Knowledge Agent", version="1.1")

UI_PATH = Path(__file__).parent / "ui.html"


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=core.TOP_K, ge=1, le=20)
    max_distance: float | None = core.MAX_DISTANCE
    category: str | None = None


class Wiadomosc(BaseModel):
    role: str
    content: str


class AskRequest(SearchRequest):
    # Historia rozmowy. Bez niej doprecyzowania w rodzaju "a dlaczego akurat tyle?"
    # albo "nastepnym razem wykorzystam to" trafialy do modelu bez punktu odniesienia
    # i dostawaly odpowiedz obok tematu.
    historia: list[Wiadomosc] = []


@app.get("/", response_class=HTMLResponse)
def interfejs() -> str:
    """Serwuje ui.html z tego samego origin - zero konfiguracji CORS."""
    if not UI_PATH.exists():
        return (
            "<h1>Brak pliku ui.html</h1>"
            f"<p>Oczekiwana lokalizacja: <code>{UI_PATH}</code></p>"
            "<p>API dziala - sprawdz <a href='/health'>/health</a>.</p>"
        )
    return UI_PATH.read_text(encoding="utf-8-sig")


@app.get("/health")
def health() -> dict:
    out = core.health()
    # parametry potrzebne interfejsowi do wyswietlenia i do oznaczania progu
    out["max_distance"] = core.MAX_DISTANCE
    out["top_k"] = core.TOP_K
    out["gate_on_best"] = core.GATE_ON_BEST
    out["use_bm25"] = core.USE_BM25
    return out


@app.post("/search")
def search(req: SearchRequest) -> dict:
    hits, all_hits = core.search(
        req.query, k=req.k, max_distance=req.max_distance, category=req.category
    )
    kept_ids = {id(h) for h in hits}
    return {
        "query": req.query,
        "in_scope": bool(hits),
        "max_distance": req.max_distance,
        "best_distance": round(all_hits[0].distance, 4) if all_hits else None,
        "hits": [h.to_dict() for h in hits],
        "rejected": [h.to_dict() for h in all_hits if id(h) not in kept_ids],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return core.answer(
        req.query,
        k=req.k,
        max_distance=req.max_distance,
        category=req.category,
        historia=[w.model_dump() for w in req.historia],
    )


@app.post("/hybrid")
def hybryda(req: AskRequest) -> dict:
    """Notatki, a gdy ich brak - wiedza modelu, jawnie oznaczona."""
    return core.answer_hybrid(
        req.query,
        k=req.k,
        max_distance=req.max_distance,
        category=req.category,
        historia=[w.model_dump() for w in req.historia],
    )


class ChatRequest(BaseModel):
    messages: list[Wiadomosc]


@app.post("/chat")
def czat(req: ChatRequest) -> dict:
    """Zwykla rozmowa z modelem, bez notatek i bez wyszukiwania."""
    t0 = time.perf_counter()
    historia = [{"role": m.role, "content": m.content} for m in req.messages]
    odpowiedz = core.rozmowa(historia)
    return {
        "answer": odpowiedz,
        "generation_s": round(time.perf_counter() - t0, 3),
        "tryb": "rozmowa",
    }


# ===================== stan modeli =====================


@app.get("/modele")
def modele() -> dict:
    return core.stan_modeli()


class PrzelaczModel(BaseModel):
    nazwa: str
    wlacz: bool


@app.post("/modele/przelacz")
def przelacz(req: PrzelaczModel) -> dict:
    wynik = core.przelacz_model(req.nazwa, req.wlacz)
    wynik["stan"] = core.stan_modeli()
    return wynik


# ===================== dopisywanie do notatek =====================


class DopiszRequest(BaseModel):
    tekst: str
    tytul: str | None = None
    reindeksuj: bool = True


@app.post("/dopisz")
def dopisz(req: DopiszRequest) -> dict:
    """Dopisuje fragment do inboxu vaultu i domyslnie od razu doindeksowuje."""
    try:
        wynik = pamiec.dopisz(req.tekst, req.tytul)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.reindeksuj:
        wynik["indeks"] = pamiec.reindeksuj()
        core.reset_cache()          # kolekcja urosla - indeks BM25 jest nieaktualny
        wynik["chunks"] = core.liczba_fragmentow()
    return wynik


@app.post("/reindeks")
def reindeks() -> dict:
    wynik = pamiec.reindeksuj()
    core.reset_cache()
    wynik["chunks"] = core.liczba_fragmentow()
    return wynik


# ===================== wpadki (zle odpowiedzi) =====================


class WpadkaRequest(BaseModel):
    pytanie: str
    odpowiedz: str = ""
    tryb: str = ""
    co_bylo_zle: str = ""
    zrodlo_wiedzy: str = ""
    max_distance: float | None = None
    k: int | None = None
    najlepszy_dystans: float | None = None
    zrodla: list[str] = []


@app.post("/wpadka")
def wpadka(req: WpadkaRequest) -> dict:
    try:
        return pamiec.zglos_wpadke(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/wpadki")
def wpadki() -> dict:
    lista = pamiec.lista_wpadek()
    return {"razem": len(lista), "wpadki": lista}


# ===================== historia rozmow =====================


class ZapiszRozmowe(BaseModel):
    id: str | None = None
    tryb: str = ""
    wpisy: list[dict]


@app.get("/rozmowy")
def rozmowy() -> dict:
    return {"rozmowy": pamiec.lista_rozmow()}


@app.get("/rozmowy/{rozmowa_id}")
def rozmowa(rozmowa_id: str) -> dict:
    try:
        return pamiec.wczytaj_rozmowe(rozmowa_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="nie ma takiej rozmowy") from exc


@app.post("/rozmowy")
def zapisz_rozmowe(req: ZapiszRozmowe) -> dict:
    try:
        return pamiec.zapisz_rozmowe(req.id, req.wpisy, req.tryb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/rozmowy/{rozmowa_id}")
def usun_rozmowe(rozmowa_id: str) -> dict:
    return {"usunieta": pamiec.usun_rozmowe(rozmowa_id)}


# ===================== zrzut sesji =====================


@app.post("/zrzut")
def zrzut() -> dict:
    return pamiec.zrzut("recznie")


@app.on_event("shutdown")
def zrzut_przy_zamknieciu() -> None:
    """Drugi z dwoch momentow zrzutu - przy zatrzymywaniu serwera.

    Zadziala przy Ctrl+C i przy normalnym zamknieciu okna. NIE zadziala, gdy
    proces zostanie ubity twardo (taskkill /F) - dlatego przycisk w interfejsie
    zostaje jako sciezka pewna.
    """
    try:
        w = pamiec.zrzut("zamkniecie serwera")
        if w.get("ok"):
            print(f"\n  Zrzut sesji: {w['plik']}")
    except Exception as exc:  # noqa: BLE001 - blad zrzutu nie moze blokowac wyjscia
        print(f"\n  Zrzut sesji nieudany: {exc}")
