# from redis import Redis
# from langchain.memory import ConversationBufferMemory
# from langchain_redis import RedisChatMessageHistory
# from cp_genie.core.config import Settings
# from langchain_community.chat_message_histories import ChatMessageHistory

# settings = Settings()
# redis = Redis.from_url(settings.redis_url, decode_responses=True)

# def get_memory(session_id: str) -> ConversationBufferMemory:
#     # history = RedisChatMessageHistory(
#     #     session_id=session_id,
#     #     redis_client=redis,
#     # )
#     return ConversationBufferMemory(
#         memory_key="chat_history",
#         chat_memory=ChatMessageHistory(),
#         return_messages=True,
#     )
