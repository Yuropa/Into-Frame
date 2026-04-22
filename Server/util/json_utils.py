import json

def parse_json(raw: str):
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}\nRaw response: {repr(raw)}")
        raise