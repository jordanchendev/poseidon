"""The one exception the research tools raise for an input they refuse (config, document, narrative, claim).

Library code raises ResearchRefusal; only a command-line entry point turns it into an exit (``sys.exit(str(e))``), and
the runner turns it into gate reasons or a needs_review reason. ``reason`` is the headline, ``problems`` the itemised
one-line reasons; ``str(e)`` is the headline followed by one indented line per problem.
"""

from __future__ import annotations

from collections.abc import Iterable


class ResearchRefusal(Exception):
    def __init__(self, reason: str, problems: Iterable[str] = ()) -> None:
        self.reason = reason
        self.problems = list(problems)
        super().__init__(reason + "".join("\n  " + p for p in self.problems))
