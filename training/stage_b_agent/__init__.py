"""Stage B: Bilingual capability/agent adapter on gpt-oss-120b.

Stage B starts from openai/gpt-oss-120b BASE, NOT from Stage A weights.
Stage A exists only to produce the synthetic TVL dataset. The adapter
produced by Stage B is the final shipping artifact.

Submodules:
    build_mix      - Build mixed training dataset (EN + synthetic TVL + anchor)
    train          - LoRA fine-tuning on Tinker
    eval           - Translation regression, capability smoke, preservation metrics
    tooling_modes  - Safe vs native tool-call formatting
"""
