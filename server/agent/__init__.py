"""The agent layer over the pipeline (docs/agentic-scene-graph-plan.md §3, §4).

knobs.yaml    the edit vocabulary: graph attributes the agent may change and the config
              keys or plan fields each one drives, with ranges and the stage to rerun from
critique.py   a vision model's structured reading of a render against the photograph,
              every issue targeted at a graph node; Anthropic or any OpenAI-compatible server
runner.py     turns graph edits into a derived config + a patched plan + `main.py run --rerun`
loop.py       perceive -> hypothesise -> build -> observe -> critique -> revise, with budgets
"""
