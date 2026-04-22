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

def write_json(obj, f=None):
    if f is not None:
        json.dump(obj, f, indent=2)
    else:
        return json.dumps(obj) + "\n"