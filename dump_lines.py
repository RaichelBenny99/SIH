import pathlib
p = pathlib.Path('services/gemini.py')
for i, line in enumerate(p.read_text().splitlines(), 1):
    if 40 <= i <= 115:
        print(f'{i:04d}: {line}')
