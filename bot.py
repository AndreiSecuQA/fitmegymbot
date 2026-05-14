import os
import io
import logging
from datetime import date
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler,
    CommandHandler, filters, ContextTypes,
)
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_missing = [v for v in ("TELEGRAM_TOKEN", "GEMINI_KEY") if not os.getenv(v)]
if _missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(_missing)}")

client = genai.Client(
    api_key=os.getenv("GEMINI_KEY"),
    http_options=types.HttpOptions(api_version="v1alpha"),
)

try:
    _all_models = [m.name for m in client.models.list()]
    logger.info("Available models: %s", _all_models)
except Exception as _e:
    logger.warning("Could not list models: %s", _e)

MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
]

PROMPTS = {
    "gym": (
        "Generate a realistic full-body image of this person in a gym setting, "
        "wearing athletic clothes, maintaining their facial features and build."
    ),
    "upper": (
        "Generate a realistic image showing only the upper body of this person in a gym, "
        "wearing a fitted athletic shirt, with visible muscle definition, maintaining their facial features."
    ),
    "30d": (
        "Generate a realistic image of this person after 30 days of consistent gym training — "
        "slightly more defined muscles, better posture, same face and body structure, "
        "in a gym setting, wearing athletic clothes."
    ),
    "60d": (
        "Generate a realistic image of this person after 60 days of consistent gym training — "
        "noticeably more muscular and defined, improved posture, same facial features, "
        "in a gym setting, wearing athletic clothes."
    ),
    "90d": (
        "Generate a realistic image of this person after 90 days of consistent gym training — "
        "significantly more muscular and lean, visible core definition, same facial features, "
        "in a gym setting, wearing athletic clothes."
    ),
}

LABELS = {
    "gym": "💪 Gym Look",
    "upper": "🏋️ Upper Body",
    "30d": "📅 +30 Days",
    "60d": "📅 +60 Days",
    "90d": "📅 +90 Days",
}

MAX_DAILY = 3
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60
USD_TO_MDL = 18.0

user_usage: dict = {}
pending_photos: dict = {}


def check_and_increment(user_id: int) -> bool:
    today = date.today()
    entry = user_usage.get(user_id)
    if entry is None or entry["date"] != today:
        user_usage[user_id] = {"count": 1, "date": today}
        return True
    if entry["count"] >= MAX_DAILY:
        return False
    entry["count"] += 1
    return True


def decrement(user_id: int):
    if user_id in user_usage:
        user_usage[user_id]["count"] = max(0, user_usage[user_id]["count"] - 1)


def estimate_cost_mdl(usage_metadata) -> float:
    try:
        input_t = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_t = getattr(usage_metadata, "candidates_token_count", 0) or 0
        cost_usd = (input_t * INPUT_PRICE_PER_M + output_t * OUTPUT_PRICE_PER_M) / 1_000_000
        return round(cost_usd * USD_TO_MDL, 4)
    except Exception:
        return 0.0


def build_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💪 Gym Look", callback_data="gym"),
            InlineKeyboardButton("🏋️ Upper Body", callback_data="upper"),
        ],
        [
            InlineKeyboardButton("📅 +30 Days", callback_data="30d"),
            InlineKeyboardButton("📅 +60 Days", callback_data="60d"),
            InlineKeyboardButton("📅 +90 Days", callback_data="90d"),
        ],
    ])


async def generate_image(photo_bytes: bytes, prompt: str):
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                ),
            )
            logger.info("Used model: %s", model)
            return response
        except Exception as e:
            logger.warning("Model %s failed: %s", model, e)
    return None


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    pending_photos[user_id] = bytes(await file.download_as_bytearray())
    await update.message.reply_text(
        "📸 Photo received! Choose what to generate:",
        reply_markup=build_keyboard(),
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    choice = query.data

    if user_id not in pending_photos:
        await query.message.reply_text("⚠️ Photo expired. Please send your photo again.")
        return

    if not check_and_increment(user_id):
        await query.message.reply_text("You've used all 3 daily generations. Come back tomorrow! 🗓")
        return

    await query.message.edit_text(f"⏳ Generating {LABELS[choice]}... Please wait.")

    photo_bytes = pending_photos.pop(user_id)

    try:
        response = await generate_image(photo_bytes, PROMPTS[choice])

        if response is None:
            decrement(user_id)
            await query.message.reply_text("❌ All models unavailable. Please try again later.")
            return

        if not response.candidates:
            decrement(user_id)
            await query.message.reply_text("🚫 Blocked by safety filters. Try a different photo.")
            return

        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                break

        if image_data is None:
            decrement(user_id)
            await query.message.reply_text("🚫 No image generated. Photo may have triggered safety filters.")
            return

        cost_mdl = estimate_cost_mdl(response.usage_metadata)
        uses_left = MAX_DAILY - user_usage[user_id]["count"]
        label = "generation" if uses_left == 1 else "generations"

        await query.message.reply_photo(
            photo=io.BytesIO(image_data),
            caption=(
                f"{LABELS[choice]}\n"
                f"💰 Cost: ~{cost_mdl:.4f} MDL\n"
                f"🔄 {uses_left} {label} left today"
            ),
        )
        await query.message.delete()

    except Exception as e:
        logger.error("Error for user %s: %s", user_id, e)
        decrement(user_id)
        msg = str(e).lower()
        if any(w in msg for w in ("safety", "blocked", "harm")):
            await query.message.reply_text("🚫 Blocked by safety filters. Please try a different photo.")
        else:
            await query.message.reply_text("❌ Something went wrong. Please try again.")
    finally:
        pending_photos.pop(user_id, None)


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_usage:
        user_usage[user_id]["count"] = 0
    await update.message.reply_text("✅ Daily limit reset. You have 3 generations available again.")


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📸 Send me a photo and I'll generate a gym version of you!")


def main():
    app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(CommandHandler("reset", handle_reset))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(~filters.PHOTO, handle_other))
    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
