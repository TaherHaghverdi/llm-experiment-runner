# LLM Positioning Runner

Run the same buyer questions across multiple model providers, with and without web search, and keep the raw answers in a traceable, resume-safe format.

This project is intentionally narrow. It collects evidence; it does not decide what your questions should be or turn the answers into a strategy. Use the companion [LLM Positioning Audit skill](https://github.com/TaherHaghverdi/llm-positioning-audit) when you want a coding agent to research your space, propose an experiment, configure this runner, and analyze the result with you.

## What it does

- Runs a JSON prompt set against OpenAI, Anthropic, and Google models.
- Supports `plain` and provider-native `web` conditions.
- Repeats each prompt so you can see whether an answer is stable or a one-off.
- Writes each completed attempt immediately to append-only JSONL.
- Can resume an interrupted or partially failed run without repeating successful jobs.
- Captures latency, token usage, search queries, cited sources, and errors when providers expose them.
- Verifies completeness against the original manifest.
- Keeps API credentials in a local `.env` file that Git ignores.

The API benchmark is a reproducible proxy. Consumer products such as ChatGPT, Claude, and Gemini can add different system instructions, memory, search behavior, and product features. Periodically compare a small sample with the consumer experiences your audience actually uses.

## Requirements

- Python 3.9 or newer
- An API key for every provider in your config

There are no runtime packages to install.

## Quick start

```bash
git clone https://github.com/TaherHaghverdi/llm-positioning-runner.git
cd llm-positioning-runner

cp .env.example .env
cp config.example.json config.json
cp prompts.example.json prompts.json
```

Add your keys to `.env`, then change `prompts_file` in `config.json` to `prompts.json`. Replace the example model IDs and prompts before spending money.

Preview the design and estimated cost:

```bash
python3 -m llm_runner plan --config config.json
```

Run it:

```bash
python3 -m llm_runner run --config config.json --env .env
```

The runner prints the run directory when it starts. Use that path to inspect or verify the run:

```bash
python3 -m llm_runner status runs/2026-07-23-example-positioning-audit-120000
python3 -m llm_runner verify runs/2026-07-23-example-positioning-audit-120000
```

If a run stops, fix the cause and resume it:

```bash
python3 -m llm_runner run \
  --config config.json \
  --env .env \
  --resume runs/2026-07-23-example-positioning-audit-120000
```

Successful jobs are skipped. Failed and unfinished jobs are attempted again.

## Configuration

`config.example.json` controls the experiment:

- `models`: provider model IDs, supported conditions, key environment variables, and optional current token and search pricing.
- `conditions`: `plain`, `web`, or both.
- `experiment.repetitions`: how many independent answers to collect for each prompt/model/condition combination.
- `request`: timeouts, retry behavior, concurrency, and output limits.
- `estimate`: assumed input/output tokens per call and searches per web-enabled call. These make the pre-run cost estimate possible.
- `budget.max_estimated_usd`: an optional guardrail. The runner refuses to start when the estimate exceeds it.

Search tools can be billed separately from model tokens. Pricing changes, so treat every price in a config as dated input rather than built-in truth. Omit pricing when you do not know it; the plan will mark its total incomplete. The runner includes the pricing snapshot in the run manifest.

The prompt file follows `schemas/prompts.schema.json`. Group prompts by buyer need or product surface so the later analysis can distinguish a broad positioning gap from one weak use case.

## Model discovery

Provider catalogs tell you which model IDs your API account can call:

```bash
python3 -m llm_runner models --provider openai --env .env
python3 -m llm_runner models --provider anthropic --env .env
python3 -m llm_runner models --provider google --env .env
```

A model being available does not mean your audience uses it. Model selection is an experiment-design decision; combine catalog discovery with current research about the assistants and models relevant to your market.

## Output contract

Each run directory contains:

- `manifest.json`: immutable experiment inputs and the complete job plan.
- `results.jsonl`: append-only attempts. A resumed job can have more than one attempt; the latest successful attempt wins.
- `status.json`: current state, heartbeat, and progress counts.

Each result records the prompt, provider, model, condition, repetition, answer text, timing, usage, estimated cost, search behavior, and any terminal error. See `schemas/result.schema.json`.

The contract version is `1.0`. Consumers should check `schema_version` instead of relying on repository history.

## Safety and privacy

- Never put API keys in JSON config, prompts, or shell commands. Use `.env` or existing environment variables.
- `.env` and `runs/` are ignored, but verify what you stage before committing.
- Treat prompt sets and raw answers as potentially sensitive business research.
- The runner redacts loaded secrets from recorded error messages, but provider responses themselves are stored as evidence.
- Use only public or explicitly authorized sources when building prompts and interpreting results.

## Development

```bash
python3 -m unittest discover -s tests -v
python3 -m llm_runner plan --config config.example.json
```

The tests use synthetic provider responses and do not make network calls.

## License

MIT
