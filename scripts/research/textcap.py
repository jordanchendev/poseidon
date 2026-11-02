"""Text trimming for notices, reasons and the JEV source window."""


def one_line(text, n):
    return cap(" ".join(str(text or "").split()), n)


def cap(text, n):
    text = str(text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"
