"""
wizualizacja.py - rzut PRAWDZIWEJ przestrzeni wektorowej na plaszczyzne.

Czyta embeddingi bezposrednio z Chromy, redukuje 1024 wymiary do dwoch metoda
PCA i generuje samodzielna strone HTML. Zadnych zmyslonych pozycji - kazda
kropka to realny fragment Twoich notatek, a odleglosci wynikaja z danych.

Mozesz nanosc na te mape wlasne pytanie: zostanie zaembedowane tym samym
modelem i zrzutowane na te sama plaszczyzne, razem z okregiem progu i podswietleniem
fragmentow, ktore trafilyby do odpowiedzi.

PCA liczona jest recznie przez numpy (SVD), zeby nie ciagnac scikit-learn -
to kilkanascie linii, a oszczedza kilkaset megabajtow zaleznosci.

WAZNE co do interpretacji: rzut z 1024 wymiarow na dwa gubi wiekszosc informacji.
Dwie kropki blisko siebie na obrazku NIE musza byc blisko w oryginalnej przestrzeni.
Mapa pokazuje strukture skupien, nie dokladne odleglosci - te sa w tabeli obok.

Uzycie:
    python wizualizacja.py
    python wizualizacja.py --pytanie "jaka baze wektorowa wybralem"
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np

import core

WYJSCIE = Path(__file__).resolve().parent / "mapa-wektorow.html"


def pobierz_dane() -> tuple[np.ndarray, list[dict]]:
    """Wyciaga z Chromy wektory razem z metadanymi."""
    kolekcja = core.get_collection(create=False)
    dane = kolekcja.get(include=["embeddings", "metadatas", "documents"])

    wektory = np.array(dane["embeddings"], dtype=np.float64)
    meta = []
    for m, d in zip(dane["metadatas"], dane["documents"]):
        m = m or {}
        rel = m.get("rel_path", m.get("source", "?"))
        meta.append(
            {
                "plik": rel,
                "kategoria": rel.replace("\\", "/").split("/")[0] if "/" in rel else "root",
                "sekcja": m.get("heading", ""),
                "tekst": " ".join((d or "").split())[:220],
            }
        )
    return wektory, meta


def pca_2d(x: np.ndarray, dodatkowy: np.ndarray | None = None):
    """Rzut na dwie pierwsze skladowe glowne. SVD zamiast scikit-learn.

    Zwraca (wspolrzedne, funkcja_rzutujaca, procent_wyjasnionej_wariancji).
    Punkt zapytania rzutujemy TA SAMA transformacja - inaczej lezalby
    w innym ukladzie wspolrzednych i jego polozenie nic by nie znaczylo.
    """
    srodek = x.mean(axis=0)
    xc = x - srodek
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    skladowe = vt[:2]
    wsp = xc @ skladowe.T

    wariancja = (s ** 2) / (s ** 2).sum()
    procent = float(wariancja[:2].sum() * 100)

    def rzutuj(v: np.ndarray) -> np.ndarray:
        return (v - srodek) @ skladowe.T

    return wsp, rzutuj, procent


def skaluj(wsp: np.ndarray, szer: int, wys: int, margines: int = 60):
    mn, mx = wsp.min(axis=0), wsp.max(axis=0)
    zakres = np.where(mx - mn == 0, 1, mx - mn)

    def do_px(p: np.ndarray) -> tuple[float, float]:
        u = (p - mn) / zakres
        return (
            margines + float(u[0]) * (szer - 2 * margines),
            margines + float(1 - u[1]) * (wys - 2 * margines),
        )

    return do_px


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description="Mapa przestrzeni wektorowej notatek.")
    ap.add_argument("--pytanie", default=None, help="nanies pytanie na mape")
    ap.add_argument("--nie-otwieraj", action="store_true")
    a = ap.parse_args()

    print("Czytam wektory z Chromy...")
    wektory, meta = pobierz_dane()
    if len(wektory) < 3:
        print("Za malo fragmentow w bazie.")
        return 1
    print(f"  {len(wektory)} fragmentow, {wektory.shape[1]} wymiarow")

    print("Licze PCA...")
    wsp, rzutuj, procent = pca_2d(wektory)
    print(f"  dwie pierwsze skladowe wyjasniaja {procent:.1f}% wariancji")

    SZER, WYS = 900, 620
    do_px = skaluj(wsp, SZER, WYS)

    punkty = []
    for i, m in enumerate(meta):
        x, y = do_px(wsp[i])
        punkty.append({**m, "x": round(x, 1), "y": round(y, 1)})

    pytanie = None
    if a.pytanie:
        print(f"Embeduje pytanie: {a.pytanie!r}")
        wek_q = np.array(core.embed([a.pytanie])[0], dtype=np.float64)
        qx, qy = do_px(rzutuj(wek_q))

        # Dystanse liczone w PELNEJ przestrzeni, nie na rzucie. Rzut sluzy tylko
        # do pokazania ukladu - odleglosci z niego byly by nieprawdziwe.
        norma = wektory / np.linalg.norm(wektory, axis=1, keepdims=True)
        nq = wek_q / np.linalg.norm(wek_q)
        dyst = 1 - (norma @ nq)
        kolejnosc = np.argsort(dyst)

        for i in range(len(punkty)):
            punkty[i]["dystans"] = round(float(dyst[i]), 4)
        for ranga, i in enumerate(kolejnosc[: core.TOP_K]):
            punkty[int(i)]["wybrany"] = ranga + 1

        pytanie = {
            "tekst": a.pytanie,
            "x": round(qx, 1),
            "y": round(qy, 1),
            "prog": core.MAX_DISTANCE,
            "najlepszy": round(float(dyst[kolejnosc[0]]), 4),
            "w_zakresie": bool(dyst[kolejnosc[0]] <= core.MAX_DISTANCE),
            "top": [
                {
                    "plik": punkty[int(i)]["plik"],
                    "sekcja": punkty[int(i)]["sekcja"],
                    "dystans": punkty[int(i)]["dystans"],
                }
                for i in kolejnosc[: core.TOP_K]
            ],
        }

    kategorie = [k for k, _ in Counter(p["kategoria"] for p in punkty).most_common()]

    WYJSCIE.write_text(
        SZABLON.replace("__PUNKTY__", json.dumps(punkty, ensure_ascii=False))
        .replace("__PYTANIE__", json.dumps(pytanie, ensure_ascii=False))
        .replace("__KATEGORIE__", json.dumps(kategorie, ensure_ascii=False))
        .replace("__SZER__", str(SZER))
        .replace("__WYS__", str(WYS))
        .replace("__PROCENT__", f"{procent:.1f}")
        .replace("__MODEL__", html.escape(core.EMBED_MODEL))
        .replace("__WYMIARY__", str(wektory.shape[1])),
        encoding="utf-8",
    )

    print(f"\nGotowe: {WYJSCIE}")
    if not a.nie_otwieraj:
        webbrowser.open(WYJSCIE.as_uri())
    return 0


SZABLON = """<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8"><title>Mapa wektorow notatek</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2a2f3a;--text:#e6e8ec;
--muted:#8b93a3;--accent:#6ea8fe;--ok:#4ade80;--warn:#fbbf24;--err:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;display:flex;min-height:100vh}
#mapa{flex:1;position:relative}
svg{display:block;width:100%;height:100vh}
.kropka{cursor:pointer;transition:r .1s}
.kropka:hover{stroke:var(--text);stroke-width:2}
aside{width:340px;border-left:1px solid var(--line);background:var(--panel);
padding:18px;overflow-y:auto;max-height:100vh}
h1{font-size:15px;margin:0 0 4px}
.pod{font-size:12px;color:var(--muted);margin:0 0 16px}
.legenda{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px}
.kat{font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
cursor:pointer;user-select:none}
.kat.off{opacity:.3}
#info{background:var(--panel2);border:1px solid var(--line);border-radius:9px;
padding:12px;font-size:12.5px;min-height:120px}
#info .plik{font-family:ui-monospace,Consolas,monospace;color:var(--accent);font-size:12px}
#info .sek{color:var(--muted)}
#info .txt{margin-top:8px;color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:14px}
td{padding:4px 6px;border-bottom:1px solid var(--line)}
td.d{font-family:ui-monospace,Consolas,monospace;text-align:right;width:58px}
td.d.in{color:var(--ok)}td.d.out{color:var(--muted)}
.uwaga{font-size:11.5px;color:var(--muted);margin-top:16px;border-top:1px solid var(--line);
padding-top:12px}
</style></head><body>
<div id="mapa"></div>
<aside>
<h1>Mapa przestrzeni wektorowej</h1>
<p class="pod">__MODEL__ · __WYMIARY__ wymiarow zrzutowane na 2 · PCA wyjasnia __PROCENT__% wariancji</p>
<div class="legenda" id="legenda"></div>
<div id="info">Najedz na kropke, zeby zobaczyc fragment.</div>
<div id="tabela"></div>
<p class="uwaga">Rzut z __WYMIARY__ wymiarow na dwa gubi wiekszosc informacji.
Kropki blisko siebie na obrazku nie musza byc blisko naprawde — mapa pokazuje
uklad skupien, a nie odleglosci. Prawdziwe dystanse sa liczone w pelnej
przestrzeni i widac je w tabeli.</p>
</aside>
<script>
const P=__PUNKTY__, Q=__PYTANIE__, KAT=__KATEGORIE__, W=__SZER__, H=__WYS__;
const BARWY=['#6ea8fe','#4ade80','#fbbf24','#f87171','#a78bfa','#22d3ee','#fb923c','#94a3b8'];
const barwa={}; KAT.forEach((k,i)=>barwa[k]=BARWY[i%BARWY.length]);
const ukryte=new Set();

function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

function rysuj(){
  let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  if(Q){
    const skala=(W-120)/4;
    s+=`<circle cx="${Q.x}" cy="${Q.y}" r="${Q.prog*skala}" fill="none"
        stroke="#f87171" stroke-width="1.5" stroke-dasharray="6 4" opacity=".55"/>`;
  }
  P.forEach((p,i)=>{
    if(ukryte.has(p.kategoria))return;
    const r=p.wybrany?7:4;
    const sw=p.wybrany?2:0;
    s+=`<circle class="kropka" data-i="${i}" cx="${p.x}" cy="${p.y}" r="${r}"
        fill="${barwa[p.kategoria]}" fill-opacity="${p.wybrany?1:.62}"
        stroke="#e6e8ec" stroke-width="${sw}"/>`;
  });
  if(Q){
    s+=`<circle cx="${Q.x}" cy="${Q.y}" r="9" fill="#0f1115" stroke="#e6e8ec" stroke-width="2.5"/>`;
    s+=`<circle cx="${Q.x}" cy="${Q.y}" r="3.5" fill="#e6e8ec"/>`;
    s+=`<text x="${Q.x}" y="${Q.y-16}" fill="#e6e8ec" font-size="12" font-weight="600"
        text-anchor="middle">pytanie</text>`;
  }
  s+='</svg>';
  document.getElementById('mapa').innerHTML=s;

  document.querySelectorAll('.kropka').forEach(el=>{
    el.addEventListener('mouseenter',()=>{
      const p=P[+el.dataset.i];
      document.getElementById('info').innerHTML=
        `<div class="plik">${esc(p.plik)}</div>`+
        (p.sekcja?`<div class="sek">&rsaquo; ${esc(p.sekcja)}</div>`:'')+
        (p.dystans!==undefined?`<div class="sek">dystans ${p.dystans}</div>`:'')+
        `<div class="txt">${esc(p.tekst)}</div>`;
    });
  });
}

document.getElementById('legenda').innerHTML=KAT.map(k=>
  `<span class="kat" data-k="${esc(k)}" style="color:${barwa[k]};border-color:${barwa[k]}44">
   ${esc(k)} (${P.filter(p=>p.kategoria===k).length})</span>`).join('');

document.getElementById('legenda').addEventListener('click',e=>{
  const el=e.target.closest('.kat'); if(!el)return;
  const k=el.dataset.k;
  if(ukryte.has(k)){ukryte.delete(k);el.classList.remove('off')}
  else{ukryte.add(k);el.classList.add('off')}
  rysuj();
});

if(Q){
  document.getElementById('tabela').innerHTML=
    `<div style="font-size:12.5px;margin-top:6px">
       <b>${esc(Q.tekst)}</b><br>
       <span style="color:${Q.w_zakresie?'#4ade80':'#fbbf24'}">
       najlepszy dystans ${Q.najlepszy} — ${Q.w_zakresie?'w zakresie':'poza progiem '+Q.prog}</span>
     </div><table>`+
    Q.top.map((t,i)=>`<tr><td>${i+1}. ${esc(t.plik)}<br>
      <span style="color:#8b93a3">${esc(t.sekcja)}</span></td>
      <td class="d ${t.dystans<=Q.prog?'in':'out'}">${t.dystans}</td></tr>`).join('')+
    '</table>';
}
rysuj();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
