from __future__ import annotations

import base64
import hashlib
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

LOGO_PATH = ASSETS_DIR / "logo.png"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
DATABASE_PATH = DATA_DIR / "fundbuero.db"

ASSETS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_IMAGE_SIZE = 5 * 1024 * 1024

CATEGORIES = [
    "Kleidung",
    "Taschen",
    "Schlüssel",
    "Elektronik",
    "Schulmaterial",
    "Trinkflaschen",
    "Sport",
    "Schmuck",
    "Sonstiges",
]

LOCATIONS = [
    "Fundgrube",
    "Hausmeister",
    "Sekretariat",
    "Zuhause – morgen abgeben",
]

STATUSES = [
    "Neu",
    "Gefunden",
    "Abgeholt",
    "Erledigt",
]

ITEM_TYPES = {
    "found": "Gefunden",
    "lost": "Verloren",
}

PAGE_ORDER = {
    "Startseite": "home",
    "Gefunden melden": "gefunden",
    "Verloren melden": "verloren",
    "Fundgrube durchsuchen": "suche",
    "Admin-Bereich": "admin",
}


# ============================================================
# STREAMLIT-KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KATH-FINDER – Digitales Fundbüro",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

def apply_design() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', Arial, sans-serif;
        }

        .stApp {
            background: #f4f7fb;
        }

        /* ---------- SEITENLEISTE ---------- */

        [data-testid="stSidebar"] {
            background: #0e3a75;
            border-right: 5px solid #f2b705;
        }

        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.2rem 0;
        }

        /* ---------- KOPFBEREICH MIT WAPPEN ---------- */

        .main-header {
            background: #0e3a75;
            border-bottom: 6px solid #f2b705;
            border-radius: 16px;
            padding: 1.2rem 1.6rem;
            display: flex;
            align-items: center;
            gap: 1.3rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 6px 18px rgba(14, 58, 117, 0.25);
        }

        .header-logo {
            width: 84px;
            height: 84px;
            object-fit: contain;
            background: #ffffff;
            border: 3px solid #f2b705;
            border-radius: 14px;
            padding: 4px;
            flex-shrink: 0;
        }

        .header-logo-fallback {
            width: 84px;
            height: 84px;
            background: #ffffff;
            color: #0e3a75;
            border: 3px solid #f2b705;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            flex-shrink: 0;
        }

        .header-title {
            color: #ffffff;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            line-height: 1.05;
        }

        .header-subtitle {
            color: #f2b705;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.35rem;
            letter-spacing: 0.03em;
        }

        /* ---------- ÜBERSCHRIFTEN ---------- */

        .page-title {
            text-align: center;
            color: #0e3a75;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 1rem 0 1.4rem 0;
        }

        .blue-heading {
            color: #0e3a75;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 1.7rem;
            font-weight: 800;
            margin: 1.2rem 0 0.8rem 0;
        }

        .section-label {
            color: #0e3a75;
            font-size: 1.25rem;
            font-weight: 800;
            font-family: 'Montserrat', Arial, sans-serif;
            margin-top: 1.4rem;
            margin-bottom: 0.55rem;
        }

        /* ---------- STARTSEITE ---------- */

        .hero-box {
            background: #ffffff;
            border: 1px solid #d7e0ec;
            border-top: 6px solid #0e3a75;
            border-radius: 16px;
            padding: 1.6rem 2rem;
            box-shadow: 0 6px 16px rgba(14, 58, 117, 0.08);
            margin-bottom: 2rem;
        }

        .hero-box h2 {
            color: #0e3a75;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 1.9rem;
            margin-bottom: 0.5rem;
        }

        .hero-box p {
            color: #3f5065;
            font-size: 1.05rem;
            margin: 0;
        }

        /* Rote Aktionskästen: weiße Schrift auf kräftigem Rot */
        .red-action {
            background: #c8102e;
            color: #ffffff;
            border-radius: 16px;
            padding: 1.5rem 1rem;
            min-height: 145px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-shadow: 0 8px 18px rgba(200, 16, 46, 0.3);
            margin-bottom: 0.7rem;
        }

        .red-action-small {
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
        }

        .red-action-large {
            font-size: 2.3rem;
            font-weight: 800;
            font-family: 'Montserrat', Arial, sans-serif;
            line-height: 1;
            margin-top: 0.35rem;
            color: #ffffff;
        }

        /* ---------- KARTEN ---------- */

        .item-card {
            background: #ffffff;
            border: 1px solid #d7e0ec;
            border-top: 4px solid #0e3a75;
            border-radius: 14px;
            padding: 0.8rem;
            min-height: 310px;
            box-shadow: 0 5px 14px rgba(14, 58, 117, 0.1);
            margin-bottom: 1rem;
        }

        .item-card-title {
            color: #0e3a75;
            font-weight: 800;
            font-size: 1.1rem;
            margin: 0.45rem 0;
        }

        .item-card-text {
            color: #3f5065;
            font-size: 0.95rem;
            margin: 0.25rem 0;
        }

        .status-badge {
            display: inline-block;
            border-radius: 12px;
            padding: 0.3rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 700;
            background: #0e3a75;
            color: #ffffff;
        }

        /* ---------- KI-BOX ---------- */

        .ai-box {
            background: #e8f1fd;
            border: 1px solid #0e3a75;
            border-radius: 14px;
            padding: 1rem;
            margin: 1rem 0;
        }

        .ai-title {
            color: #0e3a75;
            font-weight: 800;
            font-family: 'Montserrat', Arial, sans-serif;
            font-size: 1.1rem;
            margin-bottom: 0.65rem;
        }

        /* ---------- INFO-BOX ---------- */

        .info-box {
            background: #fff8df;
            border-left: 6px solid #f2b705;
            padding: 0.9rem 1rem;
            border-radius: 8px;
            color: #5a4a12;
            font-weight: 500;
            margin: 1rem 0;
        }

        /* ---------- BUTTONS ---------- */

        div.stButton > button {
            border-radius: 14px;
            font-weight: 700;
            font-family: 'Inter', Arial, sans-serif;
            min-height: 3rem;
        }

        div.stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primaryFormSubmit"] {
            background: #c8102e;
            color: #ffffff;
            border: none;
        }

        div.stButton > button[kind="secondary"] {
            border: 2px solid #0e3a75;
            color: #0e3a75;
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            font-size: 1rem;
        }

        /* ---------- FOOTER ---------- */

        .footer {
            text-align: center;
            color: #44546a;
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 3rem;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_design()


# ============================================================
# DATENBANK
# ============================================================

def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                location TEXT,
                item_date TEXT NOT NULL,
                item_time TEXT,
                additional_information TEXT,
                handover_location TEXT,
                image_path TEXT,
                status TEXT NOT NULL DEFAULT 'Neu',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_item(
    item_type: str,
    name: str,
    description: str,
    category: str,
    location: str,
    item_date: date,
    item_time: str,
    additional_information: str,
    handover_location: str,
    image_path: str | None,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO items (
                item_type,
                name,
                description,
                category,
                location,
                item_date,
                item_time,
                additional_information,
                handover_location,
                image_path,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_type,
                name.strip(),
                description.strip(),
                category,
                location.strip(),
                item_date.isoformat(),
                item_time.strip(),
                additional_information.strip(),
                handover_location,
                image_path,
                "Gefunden" if item_type == "found" else "Neu",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def get_items(
    item_type: str | None = None,
    category: str | None = None,
    date_filter: str | None = None,
    search: str = "",
    status: str | None = None,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM items WHERE 1 = 1"
    parameters: list[Any] = []

    if item_type:
        query += " AND item_type = ?"
        parameters.append(item_type)

    if category and category != "Alle Kategorien":
        query += " AND category = ?"
        parameters.append(category)

    if status and status != "Alle Status":
        query += " AND status = ?"
        parameters.append(status)

    if search.strip():
        query += """
            AND (
                LOWER(name) LIKE ?
                OR LOWER(description) LIKE ?
                OR LOWER(location) LIKE ?
                OR LOWER(category) LIKE ?
            )
        """
        search_value = f"%{search.lower().strip()}%"
        parameters.extend([search_value] * 4)

    if date_filter:
        today = date.today()

        if date_filter == "Heute":
            query += " AND item_date = ?"
            parameters.append(today.isoformat())

        elif date_filter == "Gestern":
            query += " AND item_date = ?"
            parameters.append((today - timedelta(days=1)).isoformat())

        elif date_filter == "Letzte Woche":
            query += " AND item_date >= ?"
            parameters.append((today - timedelta(days=7)).isoformat())

    query += " ORDER BY created_at DESC"

    with get_connection() as connection:
        return connection.execute(query, parameters).fetchall()


def get_item(item_id: int) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM items WHERE id = ?",
            (item_id,),
        ).fetchone()


def update_item(
    item_id: int,
    name: str,
    description: str,
    category: str,
    location: str,
    item_date: date,
    item_time: str,
    additional_information: str,
    handover_location: str,
    status: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE items
            SET
                name = ?,
                description = ?,
                category = ?,
                location = ?,
                item_date = ?,
                item_time = ?,
                additional_information = ?,
                handover_location = ?,
                status = ?
            WHERE id = ?
            """,
            (
                name.strip(),
                description.strip(),
                category,
                location.strip(),
                item_date.isoformat(),
                item_time.strip(),
                additional_information.strip(),
                handover_location,
                status,
                item_id,
            ),
        )
        connection.commit()


def delete_item(item_id: int) -> None:
    item = get_item(item_id)

    if item and item["image_path"]:
        image_path = Path(item["image_path"])
        if image_path.exists():
            image_path.unlink(missing_ok=True)

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM items WHERE id = ?",
            (item_id,),
        )
        connection.commit()


initialize_database()


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_keras_model():
    """
    Lädt das Teachable-Machine-Modell genau einmal.

    Die App läuft auch ohne Modell weiter.
    """
    if not MODEL_PATH.exists():
        return None, "Die Datei model/keras_model.h5 wurde noch nicht gefunden."

    try:
        from tensorflow import keras

        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        return model, None

    except Exception as error:
        return None, f"Das KI-Modell konnte nicht geladen werden: {error}"


@st.cache_data(show_spinner=False)
def load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        return []

    labels: list[str] = []

    try:
        content = LABELS_PATH.read_text(encoding="utf-8")

        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            # Teachable Machine labels.txt enthält häufig:
            # 0 Rucksack
            # 1 Tasche
            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    except Exception:
        return []

    return labels


def prepare_image_for_model(
    image: Image.Image,
    model: Any,
) -> np.ndarray:
    """
    Bereitet ein Bild für übliche Teachable-Machine-Keras-Modelle vor.

    Standardgröße bei Teachable Machine ist häufig 224x224.
    Falls das Modell eine andere Größe erwartet, wird sie aus
    model.input_shape ausgelesen.
    """
    target_width = 224
    target_height = 224

    try:
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) >= 3:
            if input_shape[1] is not None:
                target_height = int(input_shape[1])

            if input_shape[2] is not None:
                target_width = int(input_shape[2])

    except Exception:
        pass

    image = image.convert("RGB")
    image = image.resize((target_width, target_height))

    array = np.asarray(image).astype(np.float32)

    # Teachable Machine Keras-Modelle arbeiten üblicherweise mit
    # Werten zwischen -1 und 1.
    array = (array / 127.5) - 1.0
    array = np.expand_dims(array, axis=0)

    return array


def normalize_predictions(predictions: Any) -> np.ndarray:
    if isinstance(predictions, list):
        predictions = predictions[0]

    predictions = np.asarray(predictions).astype(np.float32)

    if predictions.ndim > 1:
        predictions = predictions[0]

    # Falls das Modell keine Wahrscheinlichkeiten ausgibt:
    if (
        np.any(predictions < 0)
        or np.sum(predictions) < 0.9
        or np.sum(predictions) > 1.1
    ):
        exp_values = np.exp(predictions - np.max(predictions))
        predictions = exp_values / np.sum(exp_values)

    return predictions


def classify_uploaded_image(
    uploaded_file,
) -> tuple[list[tuple[str, float]], str | None]:
    model, model_error = load_keras_model()

    if model is None:
        return [], model_error

    try:
        # Dateizeiger zurücksetzen, falls die Vorschau (st.image)
        # die Datei bereits ausgelesen hat.
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        image_array = prepare_image_for_model(image, model)

        predictions = model.predict(image_array, verbose=0)
        predictions = normalize_predictions(predictions)

        labels = load_labels()

        result: list[tuple[str, float]] = []

        for index, probability in enumerate(predictions):
            if index < len(labels):
                label = labels[index]
            else:
                label = f"Klasse {index + 1}"

            result.append((label, float(probability)))

        result.sort(key=lambda item: item[1], reverse=True)

        return result[:5], None

    except Exception as error:
        return [], f"Fehler bei der Bilderkennung: {error}"


def map_ai_category(label: str) -> str | None:
    """
    Ordnet KI-Klassen möglichst sinnvoll den App-Kategorien zu.
    Die echte Beschriftung aus labels.txt wird weiterhin angezeigt.
    """
    text = label.lower()

    mappings = {
        "kleidung": ["kleidung", "jacke", "pullover", "shirt", "hose", "mütze", "cap"],
        "taschen": ["tasche", "rucksack", "schulranzen", "beutel"],
        "schlüssel": ["schlüssel", "schlussel", "key"],
        "elektronik": ["elektronik", "handy", "smartphone", "tablet", "kopfhörer", "kabel"],
        "schulmaterial": ["schul", "buch", "heft", "stift", "mäppchen", "ordner"],
        "trinkflaschen": ["flasche", "trinkflasche"],
        "sport": ["sport", "ball", "schuhe", "turnbeutel"],
        "schmuck": ["schmuck", "ring", "kette", "armband"],
        "sonstiges": ["sonst", "other", "unbekannt"],
    }

    for category, keywords in mappings.items():
        if any(keyword in text for keyword in keywords):
            for available_category in CATEGORIES:
                if available_category.lower() == category:
                    return available_category

    return None


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def escape(value: Any) -> str:
    return html_escape(str(value or ""))


def html_escape(value: str) -> str:
    import html
    return html.escape(value)


def save_uploaded_image(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None

    if uploaded_file.size > MAX_IMAGE_SIZE:
        st.error("Das Bild darf höchstens 5 MB groß sein.")
        return None

    allowed_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    if uploaded_file.type not in allowed_types:
        st.error("Bitte nur JPG-, PNG- oder WEBP-Dateien hochladen.")
        return None

    try:
        # Dateizeiger zurücksetzen, falls die Vorschau (st.image)
        # die Datei bereits ausgelesen hat.
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        image.verify()
    except (UnidentifiedImageError, OSError):
        st.error("Die hochgeladene Datei ist kein gültiges Bild.")
        return None

    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    extension = allowed_types[uploaded_file.type]
    filename = f"{uuid.uuid4().hex}_{file_hash}{extension}"
    destination = UPLOAD_DIR / filename

    destination.write_bytes(uploaded_file.getvalue())

    return str(destination)


def go_to(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def render_header() -> None:
    """
    Zeigt den Schul-Kopfzeilenbereich mit Wappen auf jeder Seite.
    Ohne Wappen-Datei wird ein sauberes K-Emblem angezeigt.
    """
    if LOGO_PATH.exists():
        logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" class="header-logo" alt="Wappen">'
    else:
        logo_html = '<div class="header-logo-fallback">K</div>'

    st.markdown(
        f"""
        <div class="main-header">
            {logo_html}
            <div>
                <div class="header-title">KATH-FINDER</div>
                <div class="header-subtitle">Katharineum zu Lübeck · Digitales Fundbüro</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_red_action_button(
    small_text: str,
    large_text: str,
    page: str,
    key: str,
) -> None:
    st.markdown(
        f"""
        <div class="red-action">
            <div class="red-action-small">{small_text}</div>
            <div class="red-action-large">{large_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        f"{small_text} {large_text}",
        key=key,
        type="primary",
        use_container_width=True,
    ):
        go_to(page)


def render_item_card(item: sqlite3.Row) -> None:
    image_path = item["image_path"]

    st.markdown('<div class="item-card">', unsafe_allow_html=True)

    if image_path and Path(image_path).exists():
        st.image(image_path, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="
                height:150px;
                background:#e6edf6;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:3rem;
            ">🎒</div>
            """,
            unsafe_allow_html=True,
        )

    item_name = escape(item["name"])
    item_category = escape(item["category"])
    item_location = escape(item["location"])
    item_date = escape(item["item_date"])
    item_status = escape(item["status"])

    st.markdown(
        f"""
        <div class="item-card-title">{item_name}</div>
        <div class="item-card-text">Kategorie: {item_category}</div>
        <div class="item-card-text">Ort: {item_location}</div>
        <div class="item-card-text">Datum: {item_date}</div>
        <span class="status-badge">{item_status}</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SEITEN
# ============================================================

def render_home() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <h2>Willkommen im digitalen Fundbüro</h2>
            <p>
                Hier kannst du verlorene Gegenstände melden, gefundene Gegenstände
                eintragen und aktuelle Fundstücke durchsuchen.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        render_red_action_button(
            "ICH HABE",
            "VERLOREN",
            "verloren",
            "home_lost",
        )

    with right:
        render_red_action_button(
            "ICH HABE",
            "GEFUNDEN",
            "gefunden",
            "home_found",
        )

    st.markdown('<div class="blue-heading">Aktuelle Fundstücke</div>', unsafe_allow_html=True)

    recent_items = get_items(item_type="found", status="Gefunden")[:4]

    if not recent_items:
        st.info("Aktuell wurden noch keine Fundstücke eingetragen.")
    else:
        columns = st.columns(4)

        for column, item in zip(columns, recent_items):
            with column:
                render_item_card(item)

    st.markdown(
        """
        <div class="info-box">
            Hinweis: Gefundene Gegenstände werden bitte im Sekretariat,
            bei der Fundgrube oder beim Hausmeister abgegeben.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_found_form() -> None:
    st.markdown('<div class="page-title">Gefunden</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Bitte gib gefundene Gegenstände möglichst vollständig an.
            Wähle unbedingt aus, wo der Gegenstand abgegeben wurde.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=["jpg", "jpeg", "png", "webp"],
        key="found_image",
    )

    predictions: list[tuple[str, float]] = []
    ai_error: str | None = None

    if uploaded_file is not None:
        preview_col, prediction_col = st.columns([1, 1])

        with preview_col:
            st.image(uploaded_file, caption="Vorschau", use_container_width=True)

        with prediction_col:
            with st.spinner("Bild wird analysiert ..."):
                predictions, ai_error = classify_uploaded_image(uploaded_file)

            if predictions:
                st.markdown(
                    '<div class="ai-box"><div class="ai-title">KI-Erkennung</div>',
                    unsafe_allow_html=True,
                )

                for label, probability in predictions:
                    st.write(f"**{label}** – {probability * 100:.1f} %")

                st.markdown("</div>", unsafe_allow_html=True)

            elif ai_error:
                st.warning(
                    "Die KI ist momentan nicht verfügbar. "
                    "Die Kategorie kann trotzdem manuell ausgewählt werden."
                )
                with st.expander("Fehlerdetails anzeigen"):
                    st.write(ai_error)

    suggested_category = None

    if predictions:
        suggested_category = map_ai_category(predictions[0][0])

    default_index = 0

    if suggested_category in CATEGORIES:
        default_index = CATEGORIES.index(suggested_category)

    with st.form("found_form"):
        st.markdown('<div class="section-label">Gegenstand</div>', unsafe_allow_html=True)

        name = st.text_input(
            "Gegenstandsname *",
            placeholder="Zum Beispiel: schwarzer Rucksack",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Farbe, Marke oder besondere Merkmale",
        )

        category = st.selectbox(
            "Kategorie *",
            CATEGORIES,
            index=default_index,
        )

        column_a, column_b = st.columns(2)

        with column_a:
            location = st.text_input(
                "Fundort",
                placeholder="Zum Beispiel: Raum 204",
            )

        with column_b:
            item_date = st.date_input(
                "Funddatum *",
                value=date.today(),
            )

        item_time = st.text_input(
            "Optionale Uhrzeit",
            placeholder="Zum Beispiel: 10:30",
        )

        st.markdown('<div class="section-label">Abgegeben bei *</div>', unsafe_allow_html=True)

        handover_location = st.radio(
            "Bitte mindestens eine Möglichkeit auswählen",
            LOCATIONS,
            index=None,
        )

        submitted = st.form_submit_button(
            "Fundstück speichern",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not name.strip():
            st.error("Bitte gib einen Gegenstandsnamen ein.")

        elif handover_location is None:
            st.error("Bitte wähle aus, wo der Gegenstand abgegeben wurde.")

        else:
            saved_image = save_uploaded_image(uploaded_file)

            if uploaded_file is not None and saved_image is None:
                return

            create_item(
                item_type="found",
                name=name,
                description=description,
                category=category,
                location=location,
                item_date=item_date,
                item_time=item_time,
                additional_information="",
                handover_location=handover_location,
                image_path=saved_image,
            )

            st.success("Das Fundstück wurde erfolgreich gespeichert.")
            st.balloons()


def render_lost_form() -> None:
    st.markdown('<div class="page-title">Verloren</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Beschreibe den verlorenen Gegenstand möglichst genau.
            Ein Bild ist optional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Optionales Bild hochladen",
        type=["jpg", "jpeg", "png", "webp"],
        key="lost_image",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Vorschau", width=350)

    with st.form("lost_form"):
        name = st.text_input(
            "Gegenstandsname *",
            placeholder="Zum Beispiel: blaue Trinkflasche",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Farbe, Marke oder besondere Merkmale",
        )

        category = st.selectbox(
            "Kategorie *",
            CATEGORIES,
        )

        column_a, column_b = st.columns(2)

        with column_a:
            location = st.text_input(
                "Verlustort",
                placeholder="Zum Beispiel: Sporthalle",
            )

        with column_b:
            item_date = st.date_input(
                "Verlustdatum *",
                value=date.today(),
            )

        additional_information = st.text_area(
            "Weitere Informationen",
            placeholder="Zum Beispiel: zuletzt nach der sechsten Stunde gesehen",
        )

        submitted = st.form_submit_button(
            "Verlustmeldung speichern",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not name.strip():
            st.error("Bitte gib einen Gegenstandsnamen ein.")
            return

        saved_image = save_uploaded_image(uploaded_file)

        if uploaded_file is not None and saved_image is None:
            return

        create_item(
            item_type="lost",
            name=name,
            description=description,
            category=category,
            location=location,
            item_date=item_date,
            item_time="",
            additional_information=additional_information,
            handover_location="",
            image_path=saved_image,
        )

        st.success("Die Verlustmeldung wurde gespeichert.")


def render_search_page() -> None:
    st.markdown('<div class="page-title">Fundgrube</div>', unsafe_allow_html=True)

    st.markdown('<div class="blue-heading">Filter</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        search = st.text_input(
            "Suchbegriff",
            placeholder="Zum Beispiel Rucksack",
        )

    with col2:
        category = st.selectbox(
            "Kategorie",
            ["Alle Kategorien"] + CATEGORIES,
        )

    with col3:
        date_filter = st.selectbox(
            "Datum",
            [
                "Alle Daten",
                "Heute",
                "Gestern",
                "Letzte Woche",
            ],
        )

    col4, col5 = st.columns(2)

    with col4:
        item_type_label = st.selectbox(
            "Bereich",
            ["Alle", "Gefunden", "Verloren"],
        )

    with col5:
        status = st.selectbox(
            "Status",
            ["Alle Status"] + STATUSES,
        )

    type_mapping = {
        "Alle": None,
        "Gefunden": "found",
        "Verloren": "lost",
    }

    filter_date = None if date_filter == "Alle Daten" else date_filter

    items = get_items(
        item_type=type_mapping[item_type_label],
        category=category,
        date_filter=filter_date,
        search=search,
        status=status,
    )

    st.markdown(
        f'<div class="blue-heading">{len(items)} Ergebnisse</div>',
        unsafe_allow_html=True,
    )

    if not items:
        st.info("Keine passenden Einträge gefunden.")
        return

    columns = st.columns(3)

    for index, item in enumerate(items):
        with columns[index % 3]:
            render_item_card(item)


def render_admin_page() -> None:
    st.markdown('<div class="page-title">Admin-Bereich</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="blue-heading">Einträge verwalten</div>',
        unsafe_allow_html=True,
    )

    items = get_items()

    if not items:
        st.info("Es gibt aktuell keine Einträge.")
        return

    for item in items:
        title = f"{item['name']} · {item['category']} · {item['item_date']}"

        with st.expander(title):
            item_id = item["id"]

            with st.form(f"admin_form_{item_id}"):
                name = st.text_input("Gegenstandsname", value=item["name"])
                description = st.text_area("Beschreibung", value=item["description"] or "")

                category_index = (
                    CATEGORIES.index(item["category"])
                    if item["category"] in CATEGORIES
                    else 0
                )
                category = st.selectbox("Kategorie", CATEGORIES, index=category_index)

                location = st.text_input("Ort", value=item["location"] or "")

                try:
                    item_date_value = date.fromisoformat(item["item_date"])
                except (TypeError, ValueError):
                    item_date_value = date.today()

                item_date = st.date_input("Datum", value=item_date_value)
                item_time = st.text_input("Uhrzeit", value=item["item_time"] or "")

                additional_information = st.text_area(
                    "Weitere Informationen",
                    value=item["additional_information"] or "",
                )

                handover_index = (
                    LOCATIONS.index(item["handover_location"])
                    if item["handover_location"] in LOCATIONS
                    else 0
                )
                handover_location = st.selectbox("Abgegeben bei", LOCATIONS, index=handover_index)

                status_index = (
                    STATUSES.index(item["status"])
                    if item["status"] in STATUSES
                    else 0
                )
                status = st.selectbox("Status", STATUSES, index=status_index)

                submitted = st.form_submit_button(
                    "Änderungen speichern",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                update_item(
                    item_id=item_id,
                    name=name,
                    description=description,
                    category=category,
                    location=location,
                    item_date=item_date,
                    item_time=item_time,
                    additional_information=additional_information,
                    handover_location=handover_location,
                    status=status,
                )
                st.success("Die Änderungen wurden gespeichert.")

            if st.button("Eintrag löschen", key=f"delete_{item_id}"):
                delete_item(item_id)
                st.rerun()


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    current_page = st.session_state["page"]

    if current_page not in PAGE_ORDER.values():
        current_page = "home"

    with st.sidebar:
        selected_label = st.radio(
            "Menü",
            list(PAGE_ORDER.keys()),
            index=list(PAGE_ORDER.values()).index(current_page),
        )
        st.session_state["page"] = PAGE_ORDER[selected_label]

    # Schul-Kopfzeile mit Wappen auf jeder Seite
    render_header()

    page = st.session_state["page"]

    if page == "gefunden":
        render_found_form()
    elif page == "verloren":
        render_lost_form()
    elif page == "suche":
        render_search_page()
    elif page == "admin":
        render_admin_page()
    else:
        render_home()

    st.markdown(
        """
        <div class="footer">
            KATH-FINDER · Digitales Fundbüro · Katharineum zu Lübeck
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
