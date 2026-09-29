"""
LINE Messaging API ↔ Dify API 中継サーバー（FastAPI）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("line-dify-relay")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_API_BASE = (
    os.getenv("DIFY_API_BASE_URL")
    or os.getenv("DIFY_API_URL")
    or "https://api.dify.ai/v1"
).strip().rstrip("/")
DIFY_USER_PREFIX = os.getenv("DIFY_USER_PREFIX", "line-").strip()

if not LINE_CHANNEL_SECRET:
    logger.warning("LINE_CHANNEL_SECRET is not set")
if not LINE_CHANNEL_ACCESS_TOKEN:
    logger.warning("LINE_CHANNEL_ACCESS_TOKEN is not set")
if not DIFY_API_KEY:
    logger.warning("DIFY_API_KEY is not set")

line_configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# LINE userId -> Dify conversation_id
_conversations: dict[str, str] = {}

app = FastAPI(title="LINE-Dify Relay", version="1.2.0")


def _dify_user_id(line_user_id: str) -> str:
    if line_user_id.startswith(DIFY_USER_PREFIX):
        return line_user_id
    return f"{DIFY_USER_PREFIX}{line_user_id}"


def _extract_dify_answer(data: dict[str, Any]) -> str:
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return ""


async def call_dify_chat_async(line_user_id: str, query: str) -> str:
    """Chatbot (/chat-messages) を優先し、失敗時は Completion (/completion-messages) へフォールバック。"""
    user = _dify_user_id(line_user_id)
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    chat_url = f"{DIFY_API_BASE}/chat-messages"
    chat_payload: dict[str, Any] = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": user,
    }
    conversation_id = _conversations.get(line_user_id)
    if conversation_id:
        chat_payload["conversation_id"] = conversation_id

    completion_url = f"{DIFY_API_BASE}/completion-messages"
    completion_payload: dict[str, Any] = {
        "inputs": {"query": query},
        "response_mode": "blocking",
        "user": user,
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        logger.info(
            "Calling Dify chat-messages: %s user=%s query_len=%d",
            chat_url,
            user,
            len(query),
        )
        chat_res = await client.post(chat_url, json=chat_payload, headers=headers)
        logger.info("Dify chat-messages status: %s", chat_res.status_code)
        logger.info("Dify chat-messages body preview: %s", chat_res.text[:500])

        if chat_res.status_code == 200:
            try:
                chat_data = chat_res.json()
            except json.JSONDecodeError:
                return f"Dify応答エラー(chat): invalid JSON {chat_res.text[:200]}"
            new_conversation_id = chat_data.get("conversation_id")
            if isinstance(new_conversation_id, str) and new_conversation_id:
                _conversations[line_user_id] = new_conversation_id
            answer = _extract_dify_answer(chat_data)
            if answer:
                return answer
            return f"Dify応答受信（本文空）: {chat_data}"

        logger.warning(
            "chat-messages failed (%s): %s. Trying completion-messages.",
            chat_res.status_code,
            chat_res.text[:500],
        )

        logger.info("Calling Dify completion-messages: %s", completion_url)
        comp_res = await client.post(
            completion_url,
            json=completion_payload,
            headers=headers,
        )
        logger.info("Dify completion-messages status: %s", comp_res.status_code)
        logger.info("Dify completion-messages body preview: %s", comp_res.text[:500])

        if comp_res.status_code == 200:
            try:
                comp_data = comp_res.json()
            except json.JSONDecodeError:
                return f"Dify応答エラー(completion): invalid JSON {comp_res.text[:200]}"
            answer = _extract_dify_answer(comp_data)
            if answer:
                return answer
            return f"Dify応答受信（本文空）: {comp_data}"

        return (
            f"Dify応答エラー(chat:{chat_res.status_code}, "
            f"completion:{comp_res.status_code}): {comp_res.text[:300]}"
        )


def _reply_line_sync(reply_token: str, text: str) -> None:
    chunks: list[str] = []
    remaining = text or "（空の応答）"
    while remaining:
        chunks.append(remaining[:5000])
        remaining = remaining[5000:]
    messages = [TextMessage(text=chunk) for chunk in chunks[:5]]

    with ApiClient(line_configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages),
        )


async def reply_line_text_async(reply_token: str, text: str) -> None:
    await asyncio.to_thread(_reply_line_sync, reply_token, text)


async def handle_text_event(event: dict[str, Any]) -> None:
    reply_token = event.get("replyToken")
    message = event.get("message") or {}
    user_msg = str(message.get("text", "")).strip()
    source = event.get("source") or {}
    user_id = source.get("userId") or "default_user"

    if not reply_token:
        logger.warning("Text event without replyToken: %s", event)
        return
    if not user_msg:
        logger.info("Empty text from user %s — skipping Dify", user_id)
        await reply_line_text_async(reply_token, "メッセージを入力してください。")
        return

    logger.info("Received user message: %s from %s", user_msg, user_id)

    try:
        reply_text = await call_dify_chat_async(user_id, user_msg)
    except httpx.HTTPError as exc:
        logger.error("Dify HTTP error: %s", exc, exc_info=True)
        reply_text = f"Dify通信例外: {exc}"
    except Exception as exc:
        logger.error("Dify error: %s", exc, exc_info=True)
        reply_text = f"Dify通信例外: {exc}"

    try:
        await reply_line_text_async(reply_token, reply_text)
        logger.info("Replied to LINE successfully (len=%d)", len(reply_text))
    except Exception as exc:
        logger.error("LINE reply error: %s", exc, exc_info=True)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "line-dify-relay"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/callback")
async def callback(request: Request) -> JSONResponse:
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")
    logger.info(
        "Webhook received. signature_present=%s body_len=%d",
        bool(signature),
        len(body),
    )
    logger.debug("Webhook body: %s", body)

    try:
        data = json.loads(body) if body else {}
        events = data.get("events", [])
        if not isinstance(events, list):
            events = []
        if not events:
            logger.info("No events (verification ping)")
            return JSONResponse(status_code=200, content={"status": "verified"})
    except Exception as exc:
        logger.error("JSON Parse Error: %s", exc, exc_info=True)
        return JSONResponse(status_code=200, content={"status": "invalid_json"})

    logger.info("Processing %d event(s)", len(events))

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            logger.warning("Event[%d] is not a dict: %r", index, event)
            continue
        event_type = event.get("type")
        logger.info("Event[%d] type=%s", index, event_type)

        if event_type != "message":
            continue

        message = event.get("message") or {}
        if message.get("type") != "text":
            logger.info("Event[%d] non-text message type=%s", index, message.get("type"))
            continue

        try:
            await handle_text_event(event)
        except Exception as exc:
            logger.error("Failed to handle event[%d]: %s", index, exc, exc_info=True)

    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
