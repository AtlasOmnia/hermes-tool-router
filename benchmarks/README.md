# Router benchmark corpus

`prompts.jsonl` is a 500-record synthetic deterministic regression corpus. It verifies stable routing contracts and critical-class recall; it is not, by itself, evidence of real-world task parity.

Run:

```bash
python benchmarks/run.py
```

Release claims additionally require live Hermes full-tools-versus-routed E2E testing, provider token accounting, cache-read measurements, and a representative human-curated/adversarial corpus.
