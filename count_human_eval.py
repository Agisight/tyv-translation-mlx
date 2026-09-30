import csv
import io
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "human_eval_q4.csv"
text = open(path, encoding="utf-8-sig").read()
first = text.splitlines()[0]
delim = ";" if first.count(";") > first.count(",") else ","
rows = list(csv.reader(io.StringIO(text), delimiter=delim))
header, rows = rows[0], [r for r in rows[1:] if any(x.strip() for x in r)]
col = next(i for i, h in enumerate(header) if "оценка" in h)
scores = [r[col].strip() for r in rows if len(r) > col and r[col].strip() in ("0", "1", "2")]
c, n = Counter(scores), len(scores)
print(f"строк {len(rows)}, размечено {n}")
if n:
    for k, name in (("2", "верно"), ("1", "мелкая ошибка"), ("0", "неверно")):
        print(f"  {k} = {name}: {c[k]} ({c[k] / n:.0%})")
    print(f"  приемлемо (2 или 1): {(c['2'] + c['1']) / n:.0%}")
