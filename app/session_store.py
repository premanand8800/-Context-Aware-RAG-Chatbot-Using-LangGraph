from collections import defaultdict


class SessionStore:
    def __init__(self):
        self._messages: dict[str, list[dict[str, str]]] = defaultdict(list)

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self._messages[session_id]

    def append(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})

