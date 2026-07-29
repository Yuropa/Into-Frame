import json

def parse_json(raw: str):
    if not raw or not raw.strip():
        raise RuntimeError(f"parse_json received empty input: {repr(raw)}")
    try:
        result = json.loads(raw)
        if result is None:
            raise RuntimeError(f"parse_json got null JSON from: {repr(raw)}")
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}\nRaw response: {repr(raw)}")
        raise

def parse_json_from_stream(stdout: str):
    """Pull the last JSON object out of a command's stdout, ignoring other lines.

    For CLI helpers whose contract is "stdout is one JSON object" but that may also
    print progress to stdout. synthesize_cli did exactly that -- synthesize_pattern's
    two progress lines went to std::cout ahead of the result -- so a whole-stream
    json.loads() rejected every SUCCESSFUL call, and DistributionSynthesisStage
    silently painted nothing while the optimizer was working fine. The C++ side now
    prints those to stderr, but parsing this way means a caller no longer depends on
    the binary having been rebuilt, and no longer breaks if a future diagnostic
    reintroduces the problem.

    Returns None when stdout contains no parseable JSON object at all.
    """
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def write_json(obj, f=None):
    if f is not None:
        json.dump(obj, f, indent=2)
    else:
        return json.dumps(obj) + "\n"