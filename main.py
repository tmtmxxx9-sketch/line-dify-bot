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
from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
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


def _get_line_signature(request: Request) -> str | None:
    """X-Line-Signature を厳密に取得（大小文字の差異を吸収）。"""
    for name in ("x-line-signature", "X-Line-Signature"):
        value = request.headers.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _webhook_events_count(body: str) -> int | None:
    """本文から events 件数を推定。パース不能時は None。"""
    if not body.strip():
        return 0
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if events is None:
        return 0
    if isinstance(events, list):
        return len(events)
    return None


def _process_line_webhook(body: str, signature: str) -> None:
    handler.handle(body, signature)


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


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "line-dify-relay"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/callback")
async def line_webhook(request: Request) -> dict[str, str]:
    signature = _get_line_signature(request)
    if not signature:
        logger.warning("Missing X-Line-Signature header")
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature")

    try:
        body_bytes = await request.body()
        body = body_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("Webhook body is not valid UTF-8: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid request body") from exc

    event_count = _webhook_events_count(body)
    if event_count == 0:
        logger.info("LINE webhook with no events (verification ping or empty delivery)")

    try:
        await asyncio.to_thread(_process_line_webhook, body, signature)
    except InvalidSignatureError as exc:
        logger.warning("Invalid LINE signature: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid signature") from exc
    except json.JSONDecodeError as exc:
        logger.warning("Webhook JSON parse error after signature path: %s", exc)
    except Exception:
        logger.exception("Webhook handler error (responding 200 to LINE)")

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
