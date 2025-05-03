from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing import Sequence


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)

    def get_messages(self) -> list[BaseMessage]:
        return self.messages

    def get_history_message(self) -> list[BaseMessage] | None:
        if self.messages:
            return self.messages[:-1]
        return None

    def get_lastest_message(self) -> BaseMessage | None:
        if self.messages:
            return self.messages[-1]
        return None

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.messages.extend(messages)

    def add_user_message(self, message):
        return super().add_user_message(message)

    def add_ai_message(self, message):
        return super().add_ai_message(message)

    def clear(self) -> None:
        self.messages = []


store: dict[str, InMemoryHistory] = {}


def get_by_session_id(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
