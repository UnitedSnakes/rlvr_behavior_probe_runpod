"""Segmented, resumable, verified read-only fetch of private HF files.
Token from HF_TOKEN env (never printed). The token is sent only to huggingface.co to obtain the
signed redirect URL; byte ranges are then fetched from the signed URL without the token.
Usage: hf_fetch2.py <manifest.json> [segments]"""
import fnmatch, hashlib, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import httpx
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
SEGMENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
api = HfApi(token=TOKEN)
jobs = json.load(open(sys.argv[1]))
lock = threading.Lock()

def log(msg):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

def resolve_url(repo, rtype, rev, path):
    prefix = "datasets/" if rtype == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{repo}/resolve/{rev}/{path}"
    with httpx.Client(follow_redirects=False, timeout=60) as c:
        r = c.head(url, headers={"Authorization": f"Bearer {TOKEN}"})
        hops = 0
        while r.status_code in (301, 302, 303, 307, 308) and hops < 5:
            loc = r.headers["location"]
            if loc.startswith("/"):
                loc = "https://huggingface.co" + loc
            if "huggingface.co" in httpx.URL(loc).host:
                r = c.head(loc, headers={"Authorization": f"Bearer {TOKEN}"})
                url = loc
                hops += 1
                continue
            return loc, False  # signed CDN URL: no token needed
        return url, True  # served directly by huggingface.co

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

def get_range(task, start, end, part):
    repo, rtype, rev, rel = task[:4]
    for attempt in range(80):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if start + have > end:
            return True
        try:
            url, needs_token = resolve_url(repo, rtype, rev, rel)
            headers = {"Range": f"bytes={start + have}-{end}"}
            if needs_token:
                headers["Authorization"] = f"Bearer {TOKEN}"
            with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(60.0, connect=30.0)) as c:
                with c.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 200 and start == 0 and have == 0:
                        pass  # server ignored Range for a whole-file request: full body follows
                    elif resp.status_code != 206:
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    with open(part, "ab") as out:
                        for chunk in resp.iter_bytes(1 << 20):
                            out.write(chunk)
        except Exception as exc:
            if attempt % 5 == 4:
                log(f"segment retry {rel} [{start}] attempt {attempt}: {type(exc).__name__}")
            time.sleep(min(20, 1 + attempt))
    return (os.path.getsize(part) if os.path.exists(part) else 0) == end - start + 1

def fetch(task):
    repo, rtype, rev, rel, size, oid, dest = task
    final = os.path.join(dest, rel)
    if os.path.exists(final) and os.path.getsize(final) == size and (oid is None or sha256(final) == oid):
        return rel, "cached"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    nseg = SEGMENTS if size > (64 << 20) else 1
    bounds = [(k * size // nseg, (k + 1) * size // nseg - 1) for k in range(nseg)]
    parts = [f"{final}.seg{k:02d}" for k in range(nseg)]
    with ThreadPoolExecutor(max_workers=nseg) as pool:
        ok = list(pool.map(lambda kb: get_range(task, kb[1][0], kb[1][1], parts[kb[0]]), enumerate(bounds)))
    if not all(ok):
        return rel, "FAILED segments"
    tmp = final + ".assembling"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    if os.path.getsize(tmp) != size:
        return rel, f"FAILED size {os.path.getsize(tmp)}"
    if oid is not None and sha256(tmp) != oid:
        for p in parts:
            os.remove(p)
        os.remove(tmp)
        return rel, "FAILED sha256"
    os.replace(tmp, final)
    for p in parts:
        os.remove(p)
    return rel, "ok"

tasks = []
for job in jobs:
    info = api.repo_info(job["repo"], repo_type=job["type"], revision=job["rev"], files_metadata=True)
    for s in info.siblings:
        if any(fnmatch.fnmatch(s.rfilename, p) for p in job["patterns"]):
            tasks.append((job["repo"], job["type"], info.sha, s.rfilename, s.size, s.lfs.sha256 if s.lfs else None, job["local_dir"]))
tasks.sort(key=lambda t: t[4])
log(f"{len(tasks)} files, {sum(t[4] for t in tasks)/1e9:.2f} GB, segments={SEGMENTS}")
results = {}
for task in tasks:  # one file at a time; parallelism is within the file
    t0 = time.time()
    rel, status = fetch(task)
    results[f"{task[0]}:{rel}"] = {"status": status, "size": task[4], "sha256": task[5], "rev": task[2]}
    log(f"{status:8s} {task[0].split('/')[1]} {rel} {task[4]/1e6:.1f} MB {time.time()-t0:.0f}s")
json.dump(results, open(sys.argv[1].replace(".json", "_record.json"), "w"), indent=1)
bad = [k for k, v in results.items() if v["status"] not in ("ok", "cached")]
log("FETCH " + ("OK" if not bad else f"FAILED {bad}"))
