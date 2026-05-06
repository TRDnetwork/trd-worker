# trd-worker

Community GPU worker for the TRD Compute Network. Plug in your idle GPU, run TRD's website-builder agents in the background, and earn credits redeemable on TRD subscriptions.

> **Phase 2 — alpha.** Inference is currently STUBBED (workers sleep + return canned responses). Real inference ships in Phase 3 once we wire in llama.cpp / vllm.

## Install (dev)

```bash
cd trd-worker
pip install -e .
```

Requires Python 3.9+.

## Quickstart

```bash
# 1. Sign up at https://compute.trdn.io (waitlist gates registration)
# 2. Register this machine
trd-worker login --email you@example.com

# 3. Start the worker
trd-worker start
```

## Commands

| Command | Description |
|---|---|
| `trd-worker login` | Detect GPU, register with backend, save token |
| `trd-worker start` | Run daemon (heartbeat + poll + execute jobs) |
| `trd-worker status` | Show config + login state |
| `trd-worker logout` | Wipe local config |
| `trd-worker version` | Print version |

## Config location

`~/.trd-worker/config.json` (chmod 600). Holds your worker ID, auth token, and email.

## Environment overrides

- `TRD_CN_API` — override backend URL (for local backend testing)

## Earning rates (rounded)

| GPU | cr/hr |
|---|---|
| H100 80GB | 500 |
| A100 80GB | 300 |
| A100 40GB | 220 |
| RTX 4090 24GB | 150 |
| RTX 3090 24GB | 100 |
| AMD MI250 / 7900 XTX | 120 |
| Apple M3 Max/Ultra | 80 |
| Apple M2/M3 Pro | 50 |

1 credit ≈ $0.008 (pegged to AWS-equivalent compute, updated monthly).

## License

MIT
