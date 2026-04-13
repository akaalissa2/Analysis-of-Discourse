import re

def parse_cha_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    par_lines = re.findall(r"\*PAR:\s*(.*?)(?=\n\*[A-Z]+:|$)", content, re.DOTALL)

    cleaned, pauses_count, pauses_total = [], 0, 0.0

    for line in par_lines:
        pauses = re.findall(r"\((\d+\.?\d*)\)", line)
        pauses_count += len(pauses)
        pauses_total += sum(map(float, pauses))

        line = re.sub(r"\(\d+\.?\d*\)", "", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned.append(line)

    return {
        "text": " ".join(cleaned).strip(),
        "pauses": pauses_count,
        "pause_duration": pauses_total,
        "utterances": len(cleaned),
    }
