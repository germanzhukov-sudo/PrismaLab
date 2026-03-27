"""Клавиатуры бота и константы стилей."""

from __future__ import annotations

import os
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

from prismalab.config import MINIAPP_URL
from prismalab.pack_offers import _pack_offers


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _express_button_label(profile: Any | None) -> str:
    """Подпись кнопки Экспресс-фото: «(1 фото бесплатно)» только если бесплатная генерация не потрачена."""
    if profile and not getattr(profile, "free_generation_used", True):
        return "⚡️ Экспресс-фото (1 фото бесплатно)"
    return "⚡️ Экспресс-фото"


# ---------------------------------------------------------------------------
# Главное меню / Навигация
# ---------------------------------------------------------------------------

def _start_keyboard(profile: Any | None = None) -> InlineKeyboardMarkup:
    """Клавиатура экрана /start: Персона (Mini App), Быстрое фото, Тарифы, Примеры, FAQ."""
    rows: list[list[InlineKeyboardButton]] = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.extend([
        [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
        [InlineKeyboardButton("Тарифы и форматы съёмки", callback_data="pl_start_tariffs")],
        [InlineKeyboardButton("Примеры работ", callback_data="pl_start_examples")],
        [InlineKeyboardButton("А точно ли получится круто?", callback_data="pl_start_faq")],
    ])
    return InlineKeyboardMarkup(rows)


def _profile_keyboard(profile: Any) -> InlineKeyboardMarkup:
    """Клавиатура Профиля: Изменить пол, Экспресс-фото, Персона."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("Изменить пол", callback_data="pl_profile_toggle_gender")],
    ]
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.append([InlineKeyboardButton(_express_button_label(profile), callback_data="pl_profile_fast_tariffs")])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Экспресс-фото
# ---------------------------------------------------------------------------

def _fast_gender_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора пола для Быстрое фото: Женский, Мужской, Назад."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Женский", callback_data="pl_fast_gender:female"),
            InlineKeyboardButton("Мужской", callback_data="pl_fast_gender:male"),
        ],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _fast_tariff_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура «тарифы»: пакеты 5, 10, 30 + Персона + Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        *([[InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})]] if MINIAPP_URL else []),
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _fast_tariff_persona_only_keyboard() -> InlineKeyboardMarkup:
    """Только кнопка Персона (первое сообщение при 0 кредитах)."""
    rows: list[list[InlineKeyboardButton]] = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    return InlineKeyboardMarkup(rows)


def _fast_tariff_packages_keyboard(*, back_callback: str = "pl_fast_back") -> InlineKeyboardMarkup:
    """Пакеты 5, 10, 30 + Назад (без Персоны)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        [InlineKeyboardButton("Назад", callback_data=back_callback)],
    ])


def _fast_tariff_keyboard_from_profile() -> InlineKeyboardMarkup:
    """Тарифы экспресс-фото при переходе из Профиля: пакеты 5, 10, 30 + Персона + Назад -> Профиль."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        *([[InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})]] if MINIAPP_URL else []),
        [InlineKeyboardButton("Назад", callback_data="pl_profile")],
    ])


def _fast_upload_keyboard() -> InlineKeyboardMarkup:
    """Только «Назад» после «Загрузите фото» в Быстрое фото."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _fast_upload_or_change_keyboard() -> InlineKeyboardMarkup:
    """Загрузить фото, Поменять стиль."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Загрузить фото", callback_data="pl_fast_upload_photo"),
            InlineKeyboardButton("Поменять стиль", callback_data="pl_fast_change_style"),
        ],
    ])


def _payment_yookassa_keyboard(url: str, back_callback: str) -> InlineKeyboardMarkup:
    """Клавиатура экрана оплаты ЮKassa: Оплатить + Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💳 Оплатить", url=url)],
        [InlineKeyboardButton("Назад", callback_data=back_callback)],
    ])


# ---------------------------------------------------------------------------
# Стили Экспресс-фото
# ---------------------------------------------------------------------------

FAST_STYLES_MALE = [
    ("Ночной бар", "night_bar"),
    ("В костюме у окна", "suit_window"),
    ("Прогулка в парке", "park_walk"),
    ("Утренний кофе", "morning_coffee"),
    ("Лесной портрет", "forest_portrait"),
    ("Ночной клуб", "night_club"),
    ("Мастерская художника", "artist_workshop"),
    ("Силуэт на закате", "sunset_silhouette"),
    ("Байкер", "biker"),
    ("Пилот", "pilot"),
]

FAST_STYLES_FEMALE = [
    ("Свадебный образ", "wedding"),
    ("Мокрое окно", "wet_window"),
    ("Вечерний гламур", "evening_glamour"),
    ("Неоновый киберпанк", "neon_cyberpunk"),
    ("Драматический свет", "dramatic_light"),
    ("Городской нуар", "city_noir"),
    ("Студийный дым", "studio_smoke"),
    ("Чёрно-белая рефлексия", "bw_reflection"),
    ("Бальный зал", "ballroom"),
    ("Греческая королева", "greek_queen"),
    ("Мокрая рубашка", "wet_shirt"),
    ("Клеопатра", "cleopatra"),
    ("Old money", "old_money"),
    ("Лавандовое бьюти", "lavender_beauty"),
    ("Серебряная иллюзия", "silver_illusion"),
    ("Белоснежная чистота", "white_purity"),
    ("Бордовый бархат", "burgundy_velvet"),
    ("Серый кашемир", "grey_cashmere"),
    ("Чёрная сетка", "black_mesh"),
    ("Лавандовый шёлк", "lavender_silk"),
    ("Шёлковое бельё в отеле", "silk_lingerie_hotel"),
    ("Ванна с лепестками", "bath_petals"),
    ("Шампанское на балконе", "champagne_balcony"),
    ("Дождливое окно", "rainy_window"),
    ("Кофе в отеле", "coffee_hotel"),
    ("Джазовый бар", "jazz_bar"),
    ("Пикник на пледе", "picnic_blanket"),
    ("Художественная студия", "art_studio"),
    ("Уют зимнего камина", "winter_fireplace"),
]

FAST_STYLES_PER_PAGE = 8


def _fast_style_label(style_id: str) -> str:
    """Подпись стиля для Экспресс-фото; для custom возвращает «Свой запрос»."""
    if style_id == "custom":
        return "Свой запрос"
    return next((l for l, s in FAST_STYLES_MALE + FAST_STYLES_FEMALE if s == style_id), style_id)


def _fast_styles_keyboard(gender: str) -> InlineKeyboardMarkup:
    """10 стилей для Быстрое фото в зависимости от пола."""
    styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
    rows = [[InlineKeyboardButton(label, callback_data=f"pl_fast_style:{sid}")] for label, sid in styles]
    return InlineKeyboardMarkup(rows)


def _fast_style_choice_keyboard(
    gender: str,
    *,
    include_tariffs: bool = True,
    back_to_ready: bool = False,
    from_profile: bool = False,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Стили по страницам (как в Персоне) + Свой запрос + навигация + Тарифы/Назад."""
    styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
    total = len(styles)
    total_pages = max(1, (total + FAST_STYLES_PER_PAGE - 1) // FAST_STYLES_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * FAST_STYLES_PER_PAGE
    end = min(start + FAST_STYLES_PER_PAGE, total)
    page_styles = styles[start:end]

    rows = [[InlineKeyboardButton(label, callback_data=f"pl_fast_style:{sid}")] for label, sid in page_styles]
    rows.append([InlineKeyboardButton("✏️ Свой запрос", callback_data="pl_fast_style:custom")])

    # ctx: 0=main(pl_fast_back), 1=back_to_ready(pl_fast_show_ready), 2=from_profile(pl_profile)
    ctx = 2 if from_profile else (1 if back_to_ready else 0)
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("← Пред", callback_data=f"pl_fast_page:{page - 1}:{ctx}"))
    nav_buttons.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="pl_fast_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("След →", callback_data=f"pl_fast_page:{page + 1}:{ctx}"))
    if nav_buttons:
        rows.append(nav_buttons)

    if from_profile:
        back_data = "pl_profile"
    else:
        back_data = "pl_fast_show_ready" if back_to_ready else "pl_fast_back"
    if include_tariffs:
        rows.append([
            *([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})] if MINIAPP_URL else []),
            InlineKeyboardButton("Назад", callback_data=back_data),
        ])
    else:
        rows.append([InlineKeyboardButton("Назад", callback_data=back_data)])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Персона
# ---------------------------------------------------------------------------

def _persona_intro_keyboard(user_id: int = 0) -> InlineKeyboardMarkup:
    """Клавиатура вводного экрана Персоны: тарифы и Назад."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("✨ 299 руб – 5 фото", callback_data="pl_persona_buy:5")],
        [InlineKeyboardButton("✨ 599 руб – 20 фото", callback_data="pl_persona_buy:20")],
        [InlineKeyboardButton("✨ 999 руб – 40 фото", callback_data="pl_persona_buy:40")],
    ]
    rows.append([InlineKeyboardButton("Назад", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


def _persona_packs_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for offer in _pack_offers():
        pack_id = int(offer["id"])
        title = str(offer["title"])
        price = float(offer["price_rub"])
        expected_images = int(offer.get("expected_images") or 0)
        if expected_images > 0:
            label = f"{title} — {int(price)} ₽ ({expected_images} фото)"
        else:
            label = f"{title} — {int(price)} ₽"
        rows.append([InlineKeyboardButton(label, callback_data=f"pl_persona_pack_buy:{pack_id}")])
    rows.append([InlineKeyboardButton("Назад", callback_data="pl_persona_back")])
    return InlineKeyboardMarkup(rows)


def _persona_gender_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора пола для Персоны: Женский, Мужской, Назад."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Женский", callback_data="pl_persona_gender:female"),
            InlineKeyboardButton("Мужской", callback_data="pl_persona_gender:male"),
        ],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _persona_tariff_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура тарифов Персоны (создание): 599/20, 999/40, Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ 299 руб – 5 фото", callback_data="pl_persona_buy:5")],
        [InlineKeyboardButton("✨ 599 руб – 20 фото", callback_data="pl_persona_buy:20")],
        [InlineKeyboardButton("✨ 999 руб – 40 фото", callback_data="pl_persona_buy:40")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_back")],
    ])


def _persona_app_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура: кнопка Персона (Mini App) + Главное меню."""
    rows: list[list[InlineKeyboardButton]] = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


def _persona_credits_out_keyboard(*, with_express: bool = False, profile: Any | None = None) -> InlineKeyboardMarkup:
    """Клавиатура при закончившихся кредитах: тарифы, [Экспресс-фото], Главное меню."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("✨ 10 кредитов – 229 руб", callback_data="pl_persona_topup_buy:10")],
        [InlineKeyboardButton("✨ 20 кредитов – 439 руб", callback_data="pl_persona_topup_buy:20")],
        [InlineKeyboardButton("✨ 30 кредитов – 629 руб", callback_data="pl_persona_topup_buy:30")],
    ]
    if with_express:
        rows.append([InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


def _persona_recreate_confirm_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура подтверждения пересоздания: Продолжить, Главное меню."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Продолжить", callback_data="pl_persona_recreate_confirm")],
        [InlineKeyboardButton("Главное меню", callback_data="pl_persona_recreate_cancel")],
    ])


def _persona_topup_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура докупки кредитов: 10/229, 20/439, 30/629, Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ 10 кредитов – 229 руб", callback_data="pl_persona_topup_buy:10")],
        [InlineKeyboardButton("✨ 20 кредитов – 439 руб", callback_data="pl_persona_topup_buy:20")],
        [InlineKeyboardButton("✨ 30 кредитов – 629 руб", callback_data="pl_persona_topup_buy:30")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_show_credits_out")],
    ])


def _persona_topup_pay_keyboard(credits: int) -> InlineKeyboardMarkup:
    """Кнопка «Оплатить» для докупки кредитов."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Оплатить", callback_data=f"pl_persona_topup_confirm:{credits}")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_show_credits_out")],
    ])


def _persona_pay_confirm_keyboard(credits: int) -> InlineKeyboardMarkup:
    """Кнопка «Оплатить» перед переходом на платёжку."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Оплатить", callback_data=f"pl_persona_confirm_pay:{credits}")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_back")],
    ])


def _persona_rules_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура после правил: Всё понятно, погнали!"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Всё понятно, погнали!", callback_data="pl_persona_got_it")],
    ])


def _persona_training_keyboard() -> InlineKeyboardMarkup:
    """Универсальная клавиатура статуса: Проверить статус."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Проверить статус", callback_data="pl_persona_check_status")],
    ])


def _persona_upload_keyboard() -> InlineKeyboardMarkup:
    """Кнопка «Сбросить и начать заново» при загрузке фото Персоны."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Сбросить и начать заново", callback_data="pl_persona_reset_photos")],
    ])


def _persona_pack_upload_keyboard() -> InlineKeyboardMarkup:
    """Кнопки при загрузке фото для запуска пака."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Сбросить фото фотосета", callback_data="pl_persona_pack_reset_photos")],
        [InlineKeyboardButton("Назад к фотосетам", callback_data="pl_persona_packs")],
    ])


def _photoset_done_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


def _photoset_retry_keyboard(pack_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Запустить фотосет снова", callback_data=f"pl_persona_pack_retry:{int(pack_id)}")],
        [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
    ])


# ---------------------------------------------------------------------------
# Стили Персоны
# ---------------------------------------------------------------------------

PERSONA_STYLES_FEMALE = [
    ("Свадебный образ", "wedding"),
    ("Барби", "barbie"),
    ("Вечерний Гламур", "evening_glamour"),
    ("Волшебный лес", "magic_forest"),
    ("Дымка и тайна", "smoke_mystery"),
    ("Дымовая завеса", "smoke_veil"),
    ("Неоновый киберпанк", "neon_cyberpunk"),
    ("Городской переулок", "city_alley"),
    ("Утро в постели", "morning_bed"),
    ("Монахиня в клубе", "nun_club"),
    ("Задумчивый арлекин", "thoughtful_arlekin"),
    ("Чёрно-белая интимность", "bw_intimacy"),
    ("Туман и меланхолия", "fog_melancholy"),
    ("Ведьма на Хэллоуин", "halloween_witch"),
    ("Силуэт в дверном проёме", "doorway_silhouette"),
    ("Ночное окно", "night_window_smoke"),
    ("Белый фон", "white_background"),
    ("Мокрое окно", "wet_window"),
    ("Голливудская классика", "hollywood_classic"),
    ("Драматический свет", "dramatic_light"),
    ("Городской нуар", "city_noir"),
    ("Чёрно-белая рефлексия", "bw_reflection"),
    ("Ретро 50-х", "retro_50s"),
    ("Сепия fashion", "sepia_fashion"),
    ("Арт-деко у бассейна", "artdeco_pool"),
    ("Греческая королева", "greek_queen"),
    ("Воздушная фигура", "airy_figure"),
    ("Бальный зал", "ballroom"),
    ("Взгляд в душу", "soul_gaze"),
    ("Студийный дым", "studio_smoke"),
    ("Шёлковая роскошь", "silk_luxury"),
    ("Пиджак и тень", "blazer_shadow"),
    ("Клеопатра", "cleopatra"),
    ("Морской ветер", "sea_breeze"),
    ("Old money", "old_money"),
    ("Лавандовое бьюти", "lavender_beauty"),
    ("Серебряная иллюзия", "silver_illusion"),
    ("Белоснежная чистота", "white_purity"),
    ("Бордовый бархат", "burgundy_velvet"),
    ("Серый кашемир", "grey_cashmere"),
    ("Чёрная сетка", "black_mesh"),
    ("Лавандовый шёлк", "lavender_silk"),
    ("Ванна с лепестками", "bath_petals"),
    ("Дождливое окно", "rainy_window"),
    ("Джазовый бар", "jazz_bar"),
    ("Пикник на пледе", "picnic_blanket"),
    ("Художественная студия", "art_studio"),
    ("Уют зимнего камина", "winter_fireplace"),
]

PERSONA_STYLES_MALE = [
    ("Ночной бар", "night_bar"),
    ("В костюме у окна", "suit_window"),
    ("Прогулка в парке", "park_walk"),
    ("Утренний кофе", "morning_coffee"),
    ("Лесной портрет", "forest_portrait"),
    ("Ночной клуб", "night_club"),
    ("Мастерская художника", "artist_workshop"),
    ("Силуэт на закате", "sunset_silhouette"),
    ("Байкер", "biker"),
    ("Пилот", "pilot"),
    ("Библиотека одиночества", "library_solitude"),
    ("Туманный берег", "foggy_shore"),
    ("Городской спорт", "city_sport"),
    ("Радость на пляже", "beach_joy"),
    ("Силуэт в дверях", "door_silhouette"),
    ("Пианист в баре", "pianist_bar"),
    ("Свечи и бархат", "candles_velvet"),
    ("Дождливый вечер", "rainy_evening"),
    ("Ночная крыша", "night_rooftop"),
    ("Контраст теней", "shadow_contrast"),
    ("Белый фон", "white_background_male"),
    ("Дымная мистика", "smoky_mystery"),
    ("Улицы Нью-Йорка", "nyc_streets"),
    ("На рыбалке", "fishing"),
    ("Стильная лестница", "stylish_stairs"),
]

PERSONA_STYLES_PER_PAGE = 8


def _persona_styles_keyboard(gender: str, page: int = 0, user_id: int = 0) -> InlineKeyboardMarkup:
    """25 стилей для Персоны: по 8 на страницу (8+8+9), 1 кнопка в ряд, навигация стрелками."""
    styles = PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE
    total = len(styles)
    total_pages = (total + PERSONA_STYLES_PER_PAGE - 1) // PERSONA_STYLES_PER_PAGE
    page = max(0, min(page, total_pages - 1)) if total_pages else 0

    start = page * PERSONA_STYLES_PER_PAGE
    end = min(start + PERSONA_STYLES_PER_PAGE, total)
    page_styles = styles[start:end]

    rows = [[InlineKeyboardButton(label, callback_data=f"pl_persona_style:{sid}")] for label, sid in page_styles]

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("← Пред", callback_data=f"pl_persona_page:{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="pl_persona_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("След →", callback_data=f"pl_persona_page:{page + 1}"))
    if nav_buttons:
        rows.append(nav_buttons)

    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Примеры работ
# ---------------------------------------------------------------------------

def _examples_intro_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура под intro: кнопка «Смотреть примеры»."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Смотреть примеры", callback_data="pl_examples_show_albums")],
    ])


def _examples_nav_keyboard(page: int, total: int) -> InlineKeyboardMarkup:
    """Клавиатура навигации по альбомам примеров."""
    channel_url = (os.getenv("PRISMALAB_EXAMPLES_CHANNEL_URL") or "https://t.me/prismalab_styles/8").strip()
    rows: list[list[InlineKeyboardButton]] = []
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("←", callback_data=f"pl_examples_page:{page - 1}"))
    if page < total - 1:
        nav.append(InlineKeyboardButton("→", callback_data=f"pl_examples_page:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("Канал с образами", url=channel_url)])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)
