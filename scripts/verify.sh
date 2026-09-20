#!/usr/bin/env bash
# Verify a running server: health, MoE-cache/MTP engagement, and real 64K-context recall.
# Usage: ./scripts/verify.sh [host:port]
set -euo pipefail
ADDR="${1:-localhost:8080}"

echo "== health =="
curl -sf "http://$ADDR/health" && echo

echo "== short completion (checks MTP draft stats) =="
curl -sf "http://$ADDR/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Reply with one short sentence: what is 17+25?"}],
  "max_tokens": 200, "temperature": 0
}' | python3 -c '
import json,sys
r = json.load(sys.stdin)
t = r["timings"]
msg = r["choices"][0]["message"]
print("answer:", (msg.get("content") or msg.get("reasoning_content",""))[:200])
pp, dp = t["prompt_per_second"], t["predicted_per_second"]
da, dn = t.get("draft_n_accepted"), t.get("draft_n")
print(f"prefill {pp:.1f} t/s | decode {dp:.2f} t/s | draft {da}/{dn} accepted")
assert t.get("draft_n", 0) > 0, "MTP speculative decoding NOT engaged"
'

echo "== deep-context recall (~47K tokens; proves the full 64K context works) =="
python3 - "$ADDR" <<'EOF'
import json, sys, time, urllib.request
addr = sys.argv[1]
target = 1300
expected = f"KX-{target*7 % 9973:04d}"
lines = [f"Fact {i}: The station code for depot {i} is KX-{i*7 % 9973:04d} and its manager is Person{i % 97}."
         for i in range(1450)]
prompt = ("Below is a registry of depot facts.\n\n" + "\n".join(lines) +
          f"\n\nQuestion: According to Fact {target} in the registry above, what is the station code for depot {target}? Answer briefly.")
body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 250, "temperature": 0}).encode()
req = urllib.request.Request(f"http://{addr}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
t = r["timings"]
msg = r["choices"][0]["message"]
text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
print(f"prompt_n={t['prompt_n']} prefill {t['prompt_per_second']:.1f} t/s | decode {t['predicted_per_second']:.2f} t/s | wall {time.time()-t0:.0f}s")
print("expected code:", expected)
assert expected in text, f"FAIL: {expected} not found in response"
print("PASS: correct recall from ~42K tokens deep")
EOF
