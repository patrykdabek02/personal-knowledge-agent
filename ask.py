"""
ask.py - CLI do zadawania pytan bazie wiedzy. Nie wymaga n8n ani API.

Uzycie:
    python ask.py "Jaka biblioteke wybralem do bazy wektorowej i dlaczego?"
    python ask.py --search-only "chunkowanie"      # sam retrieval, bez LLM
    python ask.py --k 6 --max-distance 0.6 "..."
    python ask.py                                   # tryb interaktywny
"""

from __future__ import annotations

import argparse
import sys

import core


def print_result(result: dict, verbose: bool = False) -> None:
    print("\n" + "=" * 70)
    print(result["answer"])
    print("=" * 70)

    if result["sources"]:
        print("\nZrodla:")
        for s in result["sources"]:
            print(f"  - {s}")

    print(
        f"\n[retrieval {result['retrieval_s']}s | generacja {result['generation_s']}s | "
        f"w zakresie: {'tak' if result['in_scope'] else 'nie'}]"
    )

    if verbose or not result["in_scope"]:
        print("\nNajblizsze fragmenty (diagnostyka):")
        for h in result["hits"][:5]:
            print(f"  {h['distance']:.3f}  {h['source']}  > {h['heading'] or '-'}")


def search_only(query: str, k: int, max_distance: float) -> None:
    hits, all_hits = core.search(query, k=k, max_distance=max_distance)
    print(f"\nZapytanie: {query}")
    print(f"Prog: {max_distance}   Trafien po progu: {len(hits)}/{len(all_hits)}\n")
    for h in all_hits:
        mark = "OK " if h.distance <= max_distance else "  x"
        print(f"{mark} {h.distance:.3f}  {h.source}  > {h.metadata.get('heading') or '-'}")
        preview = h.text.replace("\n", " ")[:160]
        print(f"      {preview}...\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Zapytaj swoja lokalna baze wiedzy.")
    ap.add_argument("question", nargs="*", help="pytanie (pusto = tryb interaktywny)")
    ap.add_argument("--k", type=int, default=core.TOP_K)
    ap.add_argument("--max-distance", type=float, default=core.MAX_DISTANCE)
    ap.add_argument("--category", default=None, help="ogranicz do jednej kategorii (folderu)")
    ap.add_argument("--search-only", action="store_true", help="sam retrieval, bez LLM")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    question = " ".join(args.question).strip()

    if question:
        if args.search_only:
            search_only(question, args.k, args.max_distance)
        else:
            print_result(
                core.answer(
                    question, k=args.k, max_distance=args.max_distance, category=args.category
                ),
                args.verbose,
            )
        return 0

    # tryb interaktywny
    info = core.health()
    print(f"Baza: {info.get('chunks')} fragmentow | model: {core.CHAT_MODEL}")
    print("Pytaj. Pusta linia albo Ctrl+C konczy.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        try:
            if args.search_only:
                search_only(q, args.k, args.max_distance)
            else:
                print_result(
                    core.answer(
                        q, k=args.k, max_distance=args.max_distance, category=args.category
                    ),
                    args.verbose,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"BLAD: {exc}")


if __name__ == "__main__":
    sys.exit(main())
