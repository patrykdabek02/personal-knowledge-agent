"""
graf.py - graf podobienstwa fragmentow, odpowiednik widoku polaczen z Obsidiana.

Obsidian rysuje jawne [[odnosniki]]. W tym vaulcie ich nie ma, bo notatki
powstawaly pod wyszukiwanie wektorowe, nie pod reczne linkowanie. Naturalnym
odpowiednikiem krawedzi jest tu PODOBIENSTWO: dwa fragmenty laczymy, gdy dystans
miedzy nimi spada ponizej progu.

To NIE jest ozdoba - w odroznieniu od mapy PCA graf odpowiada na dwa konkretne
pytania diagnostyczne:

  DUPLIKATY  - bardzo krotka krawedz oznacza te sama tresc w dwoch notatkach.
               To wlasnie ta patologia sprawila, ze model cytowal notatke o CV
               zamiast notatki o agencie. W README figuruje jako znane
               ograniczenie, ale nigdy nie bylo widoczne.
  SIEROTY    - fragment bez zadnej krawedzi jest niepodobny do niczego w bazie.
               Czasem to unikat, czesciej zle napisana notatka, ktorej retrieval
               nie ma jak zaczepic.

Konsola wypisuje oba zestawienia - to jest wlasciwy produkt tego skryptu.
Strona HTML sluzy do ogladania struktury.

Uzycie:
    python graf.py                      # fragmenty, prog 0.45
    python graf.py --prog 0.32          # tylko bliskie pary = duplikaty
    python graf.py --poziom notatka     # 26 wezlow zamiast 132, blizej Obsidiana
"""

from __future__ import annotations

import argparse
import html
import re
import json
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path

import numpy as np

import core

WYJSCIE = Path(__file__).resolve().parent / "graf-notatek.html"

# Ponizej tego dystansu uznajemy pare za duplikat. Wartosc dobrana pod bge-m3:
# przy anizotropii tego modelu realne dystanse siedza w 0.33-0.72, wiec 0.25
# oznacza "praktycznie to samo zdanie innymi slowami".
PROG_DUPLIKATU = 0.25


def pobierz(poziom: str):
    kolekcja = core.get_collection(create=False)
    d = kolekcja.get(include=["embeddings", "metadatas", "documents"])
    wek = np.array(d["embeddings"], dtype=np.float64)

    etykiety, opisy = [], []
    for m, dok in zip(d["metadatas"], d["documents"]):
        m = m or {}
        rel = m.get("rel_path", m.get("source", "?"))
        etykiety.append(rel)
        opisy.append(
            {
                "plik": rel,
                "kategoria": rel.replace("\\", "/").split("/")[0] if "/" in rel else "root",
                "sekcja": m.get("heading", ""),
                "tekst": " ".join((dok or "").split())[:200],
                "tagi": [t.strip().lower() for t in re.split(r"[,;]", m.get("tags", "") or "")
                         if t.strip()],
            }
        )

    if poziom == "notatka":
        # Srednia wektorow fragmentow danej notatki. Przy wiekszej bazie graf
        # fragmentow zamienia sie w klebek - agregacja do notatek daje widok
        # naprawde bliski Obsidianowi.
        grupy: dict[str, list[int]] = defaultdict(list)
        for i, rel in enumerate(etykiety):
            grupy[rel].append(i)
        nowe_wek, nowe_opisy = [], []
        for rel, idx in sorted(grupy.items()):
            nowe_wek.append(wek[idx].mean(axis=0))
            nowe_opisy.append(
                {
                    "plik": rel,
                    "kategoria": rel.replace("\\", "/").split("/")[0] if "/" in rel else "root",
                    "sekcja": f"{len(idx)} fragmentow",
                    "tekst": opisy[idx[0]]["tekst"],
                    "tagi": opisy[idx[0]]["tagi"],
                }
            )
        return np.array(nowe_wek), nowe_opisy

    return wek, opisy


def graf_tagow(opisy: list[dict], min_par: int = 1):
    """Wezly = tagi, krawedz = wspolwystepowanie na tej samej notatce.

    To jest widok NAJBLIZSZY Obsidianowi, bo opiera sie na Twoich wlasnych
    deklaracjach z frontmattera, a nie na podobienstwie policzonym przez model.
    Krawedz znaczy tu "sam uznalem, ze te dwa tematy ida razem".

    Ograniczenie widoczne od razu w danych: 66 tagow na 28 notatek, z czego
    41 wystepuje dokladnie raz. Przy takim slowniku graf jest rzadki - to nie
    wada narzedzia, tylko sygnal, ze tagi wymagaja ujednolicenia.
    """
    # deduplikacja po notatce - fragmenty jednej notatki maja te same tagi
    per_notatka: dict[str, list[str]] = {}
    for o in opisy:
        per_notatka.setdefault(o["plik"], o.get("tagi") or [])

    licznik: dict[str, int] = defaultdict(int)
    pary: dict[tuple, int] = defaultdict(int)
    for tagi in per_notatka.values():
        tagi = sorted(set(tagi))
        for t in tagi:
            licznik[t] += 1
        for i in range(len(tagi)):
            for j in range(i + 1, len(tagi)):
                pary[(tagi[i], tagi[j])] += 1

    nazwy = sorted(licznik, key=lambda t: (-licznik[t], t))
    idx = {t: i for i, t in enumerate(nazwy)}
    wezly = [
        {
            "plik": t,
            "kategoria": "tag",
            "sekcja": f"{licznik[t]} notatek",
            "tekst": ", ".join(
                sorted(p for p, tg in per_notatka.items() if t in tg)
            )[:200],
            "tagi": [],
        }
        for t in nazwy
    ]
    # "d" jest tu odwrotnoscia sily: im czesciej razem, tym krotsza krawedz.
    # Dzieki temu ten sam kod rysujacy dziala bez zmian, a pary wystepujace
    # 3+ razy renderuja sie na czerwono jak duplikaty.
    linie = [
        {"a": idx[a], "b": idx[b], "d": round(max(0.05, 0.75 - 0.25 * n), 4)}
        for (a, b), n in pary.items()
        if n >= min_par
    ]
    return wezly, linie, licznik, pary


def raport_tagow(licznik, pary, wezly) -> None:
    samotne = [t for t, n in licznik.items() if n == 1]
    bez_par = [w["plik"] for w in wezly
               if not any(w["plik"] in p for p in pary)]

    print(f"\n=== TAGI: {len(licznik)} unikalnych ===")
    for t, n in sorted(licznik.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        print(f"  {n:3d}  {t}")

    print(f"\n=== NAJSILNIEJSZE POWIAZANIA ===")
    for (a, b), n in sorted(pary.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:2d}  {a} + {b}")

    print(f"\n=== TAGI UZYTE RAZ ({len(samotne)}) - kandydaci do scalenia ===")
    print("  " + ", ".join(sorted(samotne)[:40]))

    print(f"\n  tagow: {len(licznik)} · powiazan: {len(pary)} · "
          f"uzytych raz: {len(samotne)} · bez powiazan: {len(bez_par)}")


def macierz_dystansow(wek: np.ndarray) -> np.ndarray:
    n = wek / np.linalg.norm(wek, axis=1, keepdims=True)
    d = 1.0 - (n @ n.T)
    np.fill_diagonal(d, np.inf)   # nie laczymy wezla z samym soba
    return d


def krawedzie(d: np.ndarray, prog: float, max_na_wezel: int = 4):
    """Krawedzie ponizej progu, ale nie wiecej niz kilka na wezel.

    Bez limitu gesto upakowane skupienie robi klike i graf staje sie nieczytelny.
    Bierzemy tylko najblizszych sasiadow kazdego wezla.
    """
    out = set()
    for i in range(len(d)):
        blisko = np.argsort(d[i])[:max_na_wezel]
        for j in blisko:
            if d[i][j] <= prog:
                out.add((min(i, int(j)), max(i, int(j))))
    return sorted(out)


def raport(d: np.ndarray, opisy: list[dict], kraw) -> None:
    stopnie = defaultdict(int)
    for a, b in kraw:
        stopnie[a] += 1
        stopnie[b] += 1

    print("\n=== DUPLIKATY (dystans <= "
          f"{PROG_DUPLIKATU}) - ta sama tresc w dwoch miejscach ===")
    pary = [(d[a][b], a, b) for a, b in kraw if d[a][b] <= PROG_DUPLIKATU]
    if not pary:
        print("  brak - zadna para nie jest az tak podobna")
    for dyst, a, b in sorted(pary)[:15]:
        print(f"  {dyst:.3f}  {opisy[a]['plik']} > {opisy[a]['sekcja'][:26]}")
        print(f"         {opisy[b]['plik']} > {opisy[b]['sekcja'][:26]}")

    print("\n=== SIEROTY - fragmenty bez ani jednej krawedzi ===")
    sieroty = [i for i in range(len(opisy)) if stopnie[i] == 0]
    if not sieroty:
        print("  brak - kazdy fragment ma sasiada")
    for i in sieroty[:20]:
        naj = float(np.min(d[i]))
        print(f"  najblizszy sasiad {naj:.3f}  {opisy[i]['plik']} > {opisy[i]['sekcja'][:34]}")
    if len(sieroty) > 20:
        print(f"  ... i {len(sieroty) - 20} innych")

    print(f"\n  wezlow: {len(opisy)} · krawedzi: {len(kraw)} · sierot: {len(sieroty)}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description="Graf podobienstwa notatek.")
    ap.add_argument("--prog", type=float, default=0.45, help="maksymalny dystans krawedzi")
    ap.add_argument("--poziom", choices=("fragment", "notatka", "tag"), default="fragment")
    ap.add_argument("--koloruj", choices=("folder", "tag"), default="folder",
                    help="czym barwic wezly przy poziomie fragment/notatka")
    ap.add_argument("--sasiedzi", type=int, default=4, help="ile krawedzi na wezel")
    ap.add_argument("--nie-otwieraj", action="store_true")
    a = ap.parse_args()

    print(f"Czytam bazę (poziom: {a.poziom})...")
    wek, opisy = pobierz("notatka" if a.poziom == "tag" else a.poziom)
    if len(wek) < 3:
        print("Za malo danych.")
        return 1

    if a.poziom == "tag":
        wezly_raw, linie, licznik, pary = graf_tagow(opisy)
        if len(wezly_raw) < 2:
            print("Brak tagow w bazie - dodaj je we frontmatterze notatek.")
            return 1
        print(f"  {len(wezly_raw)} tagow")
        raport_tagow(licznik, pary, wezly_raw)
        wezly = [{**w, "id": i} for i, w in enumerate(wezly_raw)]
        kategorie = ["tag"]
    else:
        print(f"  {len(wek)} wezlow")
        d = macierz_dystansow(wek)
        kraw = krawedzie(d, a.prog, a.sasiedzi)
        raport(d, opisy, kraw)

        if a.koloruj == "tag":
            # pierwszy tag notatki jako kategoria - pokazuje powiazania
            # przecinajace granice folderow, ktorych podzial katalogowy nie widzi
            for o in opisy:
                o["kategoria"] = (o.get("tagi") or ["bez tagu"])[0]

        wezly = [{**o, "id": i} for i, o in enumerate(opisy)]
        linie = [{"a": a_, "b": b_, "d": round(float(d[a_][b_]), 4)} for a_, b_ in kraw]
        kategorie = sorted({o["kategoria"] for o in opisy})

    WYJSCIE.write_text(
        SZABLON.replace("__WEZLY__", json.dumps(wezly, ensure_ascii=False))
        .replace("__LINIE__", json.dumps(linie, ensure_ascii=False))
        .replace("__KATEGORIE__", json.dumps(kategorie, ensure_ascii=False))
        .replace("__PROG__", str(a.prog))
        .replace("__POZIOM__", html.escape(a.poziom))
        .replace("__PROGDUP__", str(PROG_DUPLIKATU)),
        encoding="utf-8",
    )
    print(f"\nGotowe: {WYJSCIE}")
    if not a.nie_otwieraj:
        webbrowser.open(WYJSCIE.as_uri())
    return 0


SZABLON = """<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8"><title>Graf notatek</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2a2f3a;--text:#e6e8ec;
--muted:#8b93a3;--ok:#4ade80;--warn:#fbbf24;--err:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;display:flex;height:100vh;overflow:hidden}
#plotno{flex:1}
svg{width:100%;height:100vh;cursor:grab}
svg:active{cursor:grabbing}
aside{width:330px;border-left:1px solid var(--line);background:var(--panel);
padding:18px;overflow-y:auto}
h1{font-size:15px;margin:0 0 4px}
.pod{font-size:12px;color:var(--muted);margin:0 0 14px}
.legenda{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.kat{font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
cursor:pointer;user-select:none}
.kat.off{opacity:.3}
#info{background:var(--panel2);border:1px solid var(--line);border-radius:9px;
padding:12px;font-size:12.5px;min-height:110px}
.plik{font-family:ui-monospace,Consolas,monospace;color:#6ea8fe;font-size:12px}
.sek{color:var(--muted)}.txt{margin-top:8px;color:var(--muted);font-size:12px}
.uwaga{font-size:11.5px;color:var(--muted);margin-top:14px;border-top:1px solid var(--line);
padding-top:12px}
</style></head><body>
<div id="plotno"></div>
<aside>
<h1>Graf podobienstwa</h1>
<p class="pod">poziom: __POZIOM__ · krawedz gdy dystans &le; __PROG__</p>
<div class="legenda" id="legenda"></div>
<div id="info">Najedz na wezel. Przeciagaj, zeby rozplatac.</div>
<p class="uwaga">Czerwona krawedz = dystans &le; __PROGDUP__, czyli praktycznie
ta sama tresc w dwoch miejscach. Takie pary warto scalic - inaczej konkuruja
ze soba w wynikach wyszukiwania.<br><br>
Wezel bez krawedzi jest niepodobny do niczego w bazie. Czasem to unikat,
czesciej notatka napisana tak, ze retrieval nie ma jej jak znalezc.</p>
</aside>
<script>
const N=__WEZLY__, L=__LINIE__, KAT=__KATEGORIE__, PROGDUP=__PROGDUP__;
const BARWY=['#6ea8fe','#4ade80','#fbbf24','#f87171','#a78bfa','#22d3ee','#fb923c','#94a3b8'];
const barwa={}; KAT.forEach((k,i)=>barwa[k]=BARWY[i%BARWY.length]);
const ukryte=new Set();
const W=innerWidth-330, H=innerHeight;

N.forEach((n,i)=>{const a=2*Math.PI*i/N.length;
  n.x=W/2+Math.cos(a)*Math.min(W,H)*0.34; n.y=H/2+Math.sin(a)*Math.min(W,H)*0.34;
  n.vx=0; n.vy=0;});

const sasiedzi={}; N.forEach(n=>sasiedzi[n.id]=[]);
L.forEach(l=>{sasiedzi[l.a].push(l.b); sasiedzi[l.b].push(l.a);});

function krok(){
  // odpychanie
  for(let i=0;i<N.length;i++)for(let j=i+1;j<N.length;j++){
    const dx=N[j].x-N[i].x, dy=N[j].y-N[i].y;
    const d2=Math.max(dx*dx+dy*dy,25), f=1400/d2, d=Math.sqrt(d2);
    const ux=dx/d*f, uy=dy/d*f;
    N[i].vx-=ux; N[i].vy-=uy; N[j].vx+=ux; N[j].vy+=uy;
  }
  // przyciaganie po krawedziach - im mniejszy dystans, tym mocniej
  L.forEach(l=>{
    const A=N[l.a], B=N[l.b];
    const dx=B.x-A.x, dy=B.y-A.y, d=Math.max(Math.hypot(dx,dy),1);
    const sila=(d-70)*0.012*(1.6-l.d);
    const ux=dx/d*sila, uy=dy/d*sila;
    A.vx+=ux; A.vy+=uy; B.vx-=ux; B.vy-=uy;
  });
  // do srodka + tarcie
  N.forEach(n=>{
    n.vx+=(W/2-n.x)*0.0016; n.vy+=(H/2-n.y)*0.0016;
    n.vx*=0.86; n.vy*=0.86;
    if(n!==ciagniety){n.x+=n.vx; n.y+=n.vy;}
    n.x=Math.max(20,Math.min(W-20,n.x)); n.y=Math.max(20,Math.min(H-20,n.y));
  });
}

function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

let ciagniety=null;
const plotno=document.getElementById('plotno');

function rysuj(){
  let s=`<svg viewBox="0 0 ${W} ${H}">`;
  L.forEach(l=>{
    if(ukryte.has(N[l.a].kategoria)||ukryte.has(N[l.b].kategoria))return;
    const dup=l.d<=PROGDUP;
    s+=`<line x1="${N[l.a].x}" y1="${N[l.a].y}" x2="${N[l.b].x}" y2="${N[l.b].y}"
        stroke="${dup?'#f87171':'#2a2f3a'}" stroke-width="${dup?2:1}"
        opacity="${dup?0.9:0.55}"/>`;
  });
  N.forEach(n=>{
    if(ukryte.has(n.kategoria))return;
    const r=4+Math.min(sasiedzi[n.id].length,6);
    s+=`<circle class="w" data-id="${n.id}" cx="${n.x}" cy="${n.y}" r="${r}"
        fill="${barwa[n.kategoria]}" fill-opacity="${sasiedzi[n.id].length?0.85:0.35}"
        stroke="${sasiedzi[n.id].length?'none':'#fbbf24'}" stroke-width="1.5"/>`;
  });
  s+='</svg>';
  plotno.innerHTML=s;
}

function pokaz(n){
  document.getElementById('info').innerHTML=
    `<div class="plik">${esc(n.plik)}</div>`+
    (n.sekcja?`<div class="sek">&rsaquo; ${esc(n.sekcja)}</div>`:'')+
    `<div class="sek">${sasiedzi[n.id].length} polaczen</div>`+
    `<div class="txt">${esc(n.tekst)}</div>`;
}

plotno.addEventListener('mouseover',e=>{
  const el=e.target.closest('.w'); if(el)pokaz(N[+el.dataset.id]);
});
plotno.addEventListener('mousedown',e=>{
  const el=e.target.closest('.w'); if(el)ciagniety=N[+el.dataset.id];
});
addEventListener('mouseup',()=>ciagniety=null);
addEventListener('mousemove',e=>{
  if(!ciagniety)return;
  const r=plotno.getBoundingClientRect();
  ciagniety.x=(e.clientX-r.left)/r.width*W;
  ciagniety.y=(e.clientY-r.top)/r.height*H;
});

document.getElementById('legenda').innerHTML=KAT.map(k=>
  `<span class="kat" data-k="${esc(k)}" style="color:${barwa[k]};border-color:${barwa[k]}44">
   ${esc(k)} (${N.filter(n=>n.kategoria===k).length})</span>`).join('');
document.getElementById('legenda').addEventListener('click',e=>{
  const el=e.target.closest('.kat'); if(!el)return;
  const k=el.dataset.k;
  if(ukryte.has(k)){ukryte.delete(k);el.classList.remove('off')}
  else{ukryte.add(k);el.classList.add('off')}
});

(function petla(){krok();rysuj();requestAnimationFrame(petla)})();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
