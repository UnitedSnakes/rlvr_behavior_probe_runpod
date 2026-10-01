"""Check that every per-request vLLM seed family used in this project's GSM8K-train
evaluations is disjoint from the new families (Sam decision 1).

Index range covers the whole GSM8K train split (0..7472), not just the 256-question panel."""

N = 7473
families = {
    "pi0 bank A (seed 42)": lambda i: 42 * 100_000 + i,
    "pi0 bank B (seed 42)": lambda i: 42 * 100_000 + i + 50_000,
    "seed-42 snapshot eval / bridge": lambda i: 42 * 100_000 + i + 75_000,
}
for s in (43, 44, 45):
    families[f"seed-{s} protocol eval"] = lambda i, s=s: s * 100_000 + i + 75_000
    for b in (1, 2, 3):
        families[f"seed-{s} extra batch {b}"] = lambda i, s=s, b=b: s * 100_000 + i + 75_000 + b * 1_000_000
sets = {name: {fn(i) for i in range(N)} for name, fn in families.items()}
names = list(sets)
collisions = [
    (a, b, len(sets[a] & sets[b]))
    for k, a in enumerate(names)
    for b in names[k + 1 :]
    if sets[a] & sets[b]
]
for name in names:
    print(f"{name:34s} {min(sets[name]):>10d} .. {max(sets[name]):>10d}")
print("collisions:", collisions if collisions else "none")
raise SystemExit(1 if collisions else 0)
