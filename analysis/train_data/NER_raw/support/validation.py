from typing import Dict, List, Tuple


def validate_offsets(sample: Dict) -> bool:
    """
    Verifica se os offsets das annotations são válidos
    dentro do tamanho do texto.
    """
    text = sample["text"]
    text_len = len(text)

    for start, end, _ in sample["annotations"]:
        if not (0 <= start < end <= text_len):
            return False

    return True


def count_overlapping_spans(sample: Dict) -> int:
    """
    Conta quantos spans sobrepostos existem num CV.
    """
    spans = [(s, e) for s, e, _ in sample["annotations"]]
    overlaps = 0

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            s1, e1 = spans[i]
            s2, e2 = spans[j]

            if s1 < e2 and s2 < e1:
                overlaps += 1

    return overlaps
