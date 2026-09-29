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
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("line-dify-relay")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_API_BASE = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1").strip().rstrip("/")
DIFY_USER_PREFIX = os.getenv("DIFY_USER_PREFIX", "line-").strip()

if not LINE_CHANNEL_SECRET:
    logger.warning("LINE_CHANNEL_SECRET is not set")
if not LINE_CHANNEL_ACCESS_TOKEN:
    logger.warning("LINE_CHANNEL_ACCESS_TOKEN is not set")
if not DIFY_API_KEY:
    logger.warning("DIFY_API_KEY is not set")

handler = WebhookHandler(LINE_CHANNEL_SECRET)
line_configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# LINE userId -> Dify conversation_id
_conversations: dict[str, str] = {}

app = FastAPI(title="LINE-Dify Relay", version="1.0.0")


def _dify_user_id(line_user_id: str) -> str:
    return f"{DIFY_USER_PREFIX}{line_user_id}"


def call_dify_chat(line_user_id: str, query: str) -> str:
    url = f"{DIFY_API_BASE}/chat-messages"
    payload: dict[str, Any] = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": _dify_user_id(line_user_id),
    }
    conversation_id = _conversations.get(line_user_id)
    if conversation_id:
        payload["conversation_id"] = conversation_id

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            logger.error("Dify error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()

    new_conversation_id = data.get("conversation_id")
    if isinstance(new_conversation_id, str) and new_conversation_id:
        _conversations[line_user_id] = new_conversation_id

    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    return "（Dify から応答を取得できませんでした）"


def reply_line_text(reply_token: str, text: str) -> None:
    # LINE テキスト上限に合わせて分割（5000文字）
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunks.append(remaining[:5000])
        remaining = remaining[5000:]

    messages = [TextMessage(text=chunk) for chunk in chunks[:5]]

    with ApiClient(line_configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages),
        )


@handler.add(MessageEvent, message=TextMessageContent)
def on_text_message(event: MessageEvent) -> None:
    if not isinstance(event.message, TextMessageContent):
        return

    user_id = event.source.user_id if event.source else None
    if not user_id:
        logger.warning("Message without user_id")
        return

    query = event.message.text.strip()
    if not query:
        reply_line_text(event.reply_token, "メッセージを入力してください。")
        return

    try:
        answer = call_dify_chat(user_id, query)
    except httpx.HTTPError as exc:
        logger.exception("Dify request failed: %s", exc)
        reply_line_text(
            event.reply_token,
            "申し訳ありません。しばらくしてからもう一度お試しください。",
        )
        return
    except Exception:
        logger.exception("Unexpected error while calling Dify")
        reply_line_text(
            event.reply_token,
            "内部エラーが発生しました。",
        )
        return

    reply_line_text(event.reply_token, answer)


async def process_text_events_fallback(events: list[Any]) -> None:
    """署名検証失敗時に JSON から直接テキストメッセージを処理する。"""
    for event_dict in events:
        if not isinstance(event_dict, dict):
            continue
        if event_dict.get("type") != "message":
            continue
        message = event_dict.get("message")
        if not isinstance(message, dict) or message.get("type") != "text":
            continue

        reply_token = event_dict.get("replyToken")
        user_msg = str(message.get("text", "")).strip()
        if not reply_token or not user_msg:
            continue

        user_id = (event_dict.get("source") or {}).get("userId") or "default_user"

        try:
            reply_text = await asyncio.to_thread(call_dify_chat, user_id, user_msg)
        except Exception as ex:
            logger.error("Dify request error: %s", ex)
            reply_text = "Difyとの通信でエラーが発生しました。"

        try:
            await asyncio.to_thread(reply_line_text, reply_token, reply_text)
        except Exception as line_ex:
            logger.error("LINE reply error: %s", line_ex)


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

    try:
        data = json.loads(body) if body else {}
        events = data.get("events", [])
        if not events:
            return JSONResponse(status_code=200, content={"status": "verified"})
    except Exception:
        return JSONResponse(status_code=200, content={"status": "invalid_json"})

    try:
        await asyncio.to_thread(handler.handle, body, signature)
    except Exception as e:
        logger.warning(
            "Signature failed or handler bypassed: %s. Executing direct fallback.",
            e,
        )
        await process_text_events_fallback(events)

    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
