"""The Nemotron-H MoE engine as serve sees it (`strata-nemotron`, GGUF architecture nemotron_h_moe, e.g. NVIDIA Nemotron
3.5 Bolt 74B-A9B): one request at a time, no drafter, no images. Its arguments are the config's own; the end tokens are
the template's `<|im_end|>` and the GGUF's `</s>`."""
from __future__ import annotations

ARCH = "nemotron_h_moe"
EXE_PREFIX = "strata-nemotron"
END_TEXTS = ("<|im_end|>", "</s>")
REFUSED = {
    "logprobs": "logprobs: not supported by the Nemotron engine yet",
    "logit_bias": "logit_bias: not supported by the Nemotron engine yet",
}


def _base(exe) -> str:
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_nemotron(cfg: dict | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    return cfg.get("arch") == ARCH or _base(cfg.get("exe")).startswith(EXE_PREFIX)


def is_nemotron_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    from strata_control import user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    if "--pool-workers" in args:                      # the server's CPU thread limit, as this engine's --threads
        i = args.index("--pool-workers")
        val = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
        if val.isdigit() and int(val) > 0 and "--threads" not in args:
            args += ["--threads", val]
    if "--serve" not in args:
        args.append("--serve")
    return args


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        try:
            got = tokenizer.encode(text, parse_special=True)
        except Exception:                             # noqa: BLE001
            continue
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def refuse(engine, req: dict) -> None:
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "nemotron_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])
