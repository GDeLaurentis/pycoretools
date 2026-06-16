# imports


def split_top_level_commas(s):
    entries = []
    start = 0
    square_depth = 0

    for i, ch in enumerate(s):
        if ch == "[":
            square_depth += 1
        elif ch == "]":
            square_depth -= 1
        elif ch == "," and square_depth == 0:
            entries.append(s[start:i])
            start = i + 1

    entries.append(s[start:])
    return entries
