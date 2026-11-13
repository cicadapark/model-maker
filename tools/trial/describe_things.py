#!/usr/bin/env python3
"""A local language model asked what a thing looks like, for the picture model (docs/trials/
2026-11-11-030-diagnosis.md, "A describer for any name").

    describe_things.py URL OUT.json NAME...     URL: a llama-server (llama.cpp, MIT) serving the model

The model is asked through llama-server's chat API, its thinking off, temperature 0, with one example (a war hammer,
in the words' own style). The trial's model was Qwen3.5-9B (Apache-2.0), Q4_K_M, on the CPU."""
import json, sys, urllib.request

SYSTEM = ("You help a picture model paint a single object it may know only by name. Describe what the object looks "
          "like in one phrase of at most 35 words: its visible parts, their shapes, sizes and how they are arranged. "
          "No colours, no materials unless the name says them, no history, no use, no people, no background. "
          "Start with what the object is, then its parts.")
EXAMPLE_USER = "a war hammer"
EXAMPLE_ANSWER = ("a hafted weapon: a long straight handle with a heavy blunt block of a head at its top, a flat square "
                  "striking face on one side")

def described(url, name):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": EXAMPLE_USER},
                         {"role": "assistant", "content": EXAMPLE_ANSWER}, {"role": "user", "content": name}],
            "temperature": 0.0, "seed": 0, "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.load(response)["choices"][0]["message"]["content"]
    return " ".join(answer.split()).strip().rstrip(".")

if __name__ == "__main__":
    url, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    found = {name: described(url, name) for name in names}
    json.dump(found, open(out, "w"), indent=1)
    for name, said in found.items():
        print(f"{name}: {said}", flush=True)
