"""MaaaLink (Find the Right Care) - Databricks App (Streamlit), version 4.

Flow: step 1, say what you need to do (patient now, compare options, coverage);
step 2, pick the service; step 3, say where you are starting from; then the results.

All ranking and evidence logic lives in SQL (workspace.care_finder.recommend_at).
This file only resolves the starting place and presents the results.
"""

import base64
import html
import os
import datetime
import re
from urllib.parse import quote as url_quote

import numpy as np
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

BRAND = "MaaaLink"     # from "Maa", Hindi for mother. The name promises a connection, not an outcome.
SCHEMA = "workspace.care_finder"
SEP = " ||| "  # separator used to pass arrays from SQL to Python as one string

# Colours and type from the design
PLUM, PLUM_DARK, INK, MUTED = "#5B2A4E", "#3A1A32", "#2B2330", "#4E4554"
CREAM, CREAM_RULE, YELLOW = "#FFF3D9", "#E2CF9F", "#F2B01E"
RED, RED_DARK, RED_TINT, RULE = "#B3261E", "#7A1712", "#FBE9E7", "#D9D2DC"

# Keep in sync with workspace.care_finder.service_taxonomy
# The user says who the patient is. Each answer maps to one service in the SQL tables.
# Emergency C-section is not offered on its own: it is one of the four things counted for
# a mother, because a person in an emergency cannot be expected to tell the two apart.
SERVICES = {
    "A mother": "high_risk_delivery",
    "A newborn": "nicu",
}
LABEL_OF = {v: k for k, v in SERVICES.items()}
SERVICE_TEXT = {
    "high_risk_delivery": {
        "blurb": "A pregnancy or birth emergency, including an emergency C-section",
        "title": "Birth emergency care",
        "mentions": "C-section or high-risk pregnancy care",
    },
    "nicu": {
        "blurb": "A baby born too early or very sick",
        "title": "Newborn intensive care",
        "mentions": "newborn care",
    },
}
# Shown as switched-off cards. The data for the first four exists in SQL but is not offered.
COMING_SOON = ["Dialysis", "Cardiac surgery", "MRI scan", "Burn care", "Trauma care"]

# What each service is rated on: (label, column holding the quote)
COUNTED = {
    "high_risk_delivery": [("C-section", "sig_csection"), ("Blood", "sig_blood"),
                           ("24-hour emergency care", "__hours__"),
                           ("Newborn care", "sig_newborn")],
}
# Shown for information only, never counted
INFO_ROWS = {
    "high_risk_delivery": [("Says it handles high-risk pregnancy", "sig_highrisk"),
                           ("Anaesthesia or operating theatre", "sig_surgery")],
    "nicu": [("24-hour service", "__hours_any__"), ("Blood", "sig_blood")],
}
NICU_LEVELS = {
    "1_strong": ("NICU with ventilator or specialist",
                 "neonatal ventilation or surfactant, or a NICU together with a neonatologist"),
    "2_medium": ("Special newborn unit or CPAP", "a special newborn care unit or neonatal CPAP"),
    "3_weak": ("NICU named, no details", "a NICU is named, with no equipment or staff detail"),
    "4_basic": ("Basic newborn care only",
                "a radiant warmer, newborn phototherapy or a stabilisation unit, but no NICU"),
}

# The suggested first choice (patient-now view). See decide() for the calculation.
# Not reaching care after two tries is counted as this many km of travel.
PENALTY_KM = 600
# Stand-in for the chance that a newborn unit can help. For a mother the stand-in is
# items mentioned / (items + 1). Neither is a measured probability.
NICU_CHANCE = {"1_strong": 0.75, "2_medium": 0.50, "3_weak": 0.25, "4_basic": 0.0}

# ---- The card for the driver ---------------------------------------------------
# Two fixed sentences per language: "Please take me to this hospital." and "It is an
# emergency." They were written for this tool and have NOT been checked by native
# speakers. The facility's name and address are never translated or transliterated: a wrong
# letter could send someone to the wrong place.
DRIVER_SENTENCES = {
    "Hindi": ("हिन्दी", "कृपया मुझे इस अस्पताल ले चलिए।", "यह इमरजेंसी है।"),
    "Marathi": ("मराठी", "कृपया मला या रुग्णालयात घेऊन चला.", "हे तातडीचे आहे."),
    "Tamil": ("தமிழ்", "தயவுசெய்து என்னை இந்த மருத்துவமனைக்கு அழைத்துச் செல்லுங்கள்.", "இது அவசரம்."),
    "Gujarati": ("ગુજરાતી", "કૃપા કરીને મને આ હોસ્પિટલ લઈ જાઓ.", "આ ઇમરજન્સી છે."),
    "Telugu": ("తెలుగు", "దయచేసి నన్ను ఈ ఆసుపత్రికి తీసుకెళ్లండి.", "ఇది అత్యవసరం."),
    "Bengali": ("বাংলা", "দয়া করে আমাকে এই হাসপাতালে নিয়ে চলুন।", "এটা জরুরি।"),
    "Kannada": ("ಕನ್ನಡ", "ದಯವಿಟ್ಟು ನನ್ನನ್ನು ಈ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದುಕೊಂಡು ಹೋಗಿ.", "ಇದು ತುರ್ತು ಪರಿಸ್ಥಿತಿ."),
    "Punjabi": ("ਪੰਜਾਬੀ", "ਕਿਰਪਾ ਕਰਕੇ ਮੈਨੂੰ ਇਸ ਹਸਪਤਾਲ ਲੈ ਚੱਲੋ।", "ਇਹ ਐਮਰਜੈਂਸੀ ਹੈ।"),
    "Malayalam": ("മലയാളം", "ദയവായി എന്നെ ഈ ആശുപത്രിയിലേക്ക് കൊണ്ടുപോകൂ.", "ഇത് അടിയന്തരമാണ്."),
    "English": ("English", "Please take me to this hospital.", "It is an emergency."),
}
LANGUAGE_CODE = {"Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Gujarati": "gu", "Telugu": "te",
                 "Bengali": "bn", "Kannada": "kn", "Punjabi": "pa", "Malayalam": "ml",
                 "English": "en"}
# Main language of each state, chosen by us (it is not in the dataset). These nine cover
# 95.2% of the 1,262 facilities listed when this was counted (1,241 are listed now, after
# the name and wording exclusions); everywhere else the card opens in English.
STATE_LANGUAGE = {
    "maharashtra": "Marathi", "uttar pradesh": "Hindi", "gujarat": "Gujarati",
    "tamil nadu": "Tamil", "haryana": "Hindi", "karnataka": "Kannada", "rajasthan": "Hindi",
    "west bengal": "Bengali", "punjab": "Punjabi", "bihar": "Hindi", "kerala": "Malayalam",
    "telangana": "Telugu", "madhya pradesh": "Hindi", "delhi": "Hindi",
    "andhra pradesh": "Telugu", "jharkhand": "Hindi", "chhattisgarh": "Hindi",
    "uttarakhand": "Hindi", "himachal pradesh": "Hindi",
}
# The state field is messy: misspellings, and district or city names in place of the state.
# These are the variants found among the listed facilities (56 facilities in 44 spellings).
STATE_FIX = {
    "tamilnadu": "tamil nadu", "chennai": "tamil nadu", "erode": "tamil nadu",
    "kanchipuram": "tamil nadu", "namakkal": "tamil nadu", "salem": "tamil nadu",
    "dhule": "maharashtra", "mumbai": "maharashtra", "thane": "maharashtra",
    "latur": "maharashtra", "mh": "maharashtra", "nashik": "maharashtra",
    "navi mumbai": "maharashtra", "pune": "maharashtra",
    "satara district, maharashtra": "maharashtra",
    "hyderabad": "telangana", "indore": "madhya pradesh",
    "kushinagar": "uttar pradesh", "up": "uttar pradesh", "ghaziabad": "uttar pradesh",
    "uttar prades h": "uttar pradesh",
    "thiruvananthapuram": "kerala", "alappuzha": "kerala", "kannur": "kerala",
    "kollam": "kerala", "ajmer": "rajasthan", "barmer": "rajasthan",
    "chattisgarh": "chhattisgarh", "dakshin kannad": "karnataka",
    "ramanagara district, karnataka": "karnataka",
    "haridwar, uttarakhand": "uttarakhand", "uttaranchal": "uttarakhand",
    "hooghly": "west bengal", "hoogly": "west bengal", "uttar dinajpur": "west bengal",
    "jalandhar": "punjab", "ludhiana": "punjab", "mohali": "punjab", "kamrup": "assam",
    "jammu & kashmir": "jammu and kashmir", "kashmir": "jammu and kashmir",
    "srinagar kashmir": "jammu and kashmir", "nct delhi": "delhi", "west delhi": "delhi",
}
# (number, what it is). Sources: india.gov.in helpline directory (112, 102) and answers of
# the Ministry of Health and Family Welfare in the Lok Sabha (102, 104, 108).
EMERGENCY_NUMBERS = [
    ("112", "National emergency number"),
    ("102", "Ambulance, mainly for pregnant women and sick children"),
    ("108", "Ambulance, mainly for critical and trauma cases"),
]

# ---- Questions about the data (Databricks Genie) ----------------------------------
# The Genie agent is attached to the app as a resource; app.yaml passes its id here.
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
GENIE_WAIT_MINUTES = 2
# Each of these was asked in Genie and the numbers were checked against our own queries.
GENIE_EXAMPLES = [
    "How many facilities have a record that mentions newborn intensive care, by level?",
    "How many facilities are listed for birth emergency care, by rating?",
    "Which 10 states have the most listed facilities for either service?",
]

# ---- A name that points to another specialty ----------------------------------------
# Some records carry text from a different facility: an eye hospital whose record says
# "10-bed NICU". The sentence itself looks fine, so the only clue is the name. A facility
# whose name points to another specialty, and to nothing maternal or general, gets a
# warning and is never suggested as a first choice or backup. It stays on every list.
OTHER_SPECIALTY = [
    (r"\beye\b|eyetec|\bnetra|ophthal", "an eye hospital"),
    (r"dental|dentist", "a dental clinic"),
    (r"ortho", "an orthopaedic centre"),
    (r"kidney|dialysis|nephro|urolog", "a kidney centre"),
    (r"\bheart\b|cardiac|cardio", "a heart hospital"),
    (r"cancer|oncolog", "a cancer hospital"),
    (r"\bskin\b|dermat", "a skin clinic"),
    (r"homeo|ayurved", "a homeopathy or ayurveda clinic"),
    (r"psychiat|physio", "a clinic for another specialty"),
]
# only when the name does not also say "hospital"
OTHER_SPECIALTY_NO_HOSPITAL = [(r"\bscan\b|diagnos|patholog|laborator|imaging",
                                "a diagnostic centre")]
# words that make birth or newborn care plausible whatever else the name says
NAME_FITS = (r"matern|wom[ae]n|mother|child|p(a)?ediatr|neonat|gyn|obstet|fertil|\bivf\b|"
             r"multi|general|super.?special|medical college|medical sciences|nursing home")

# ---- The questions page ------------------------------------------------------------
# Things a record can be asked for: key -> (label, where the quoted sentence is).
ASK_ITEMS = {
    "blood": ("Blood", "sig_blood"),
    "csection": ("C-section", "sig_csection"),
    "emergency": ("24-hour emergency care", "__hours__"),
    "nicu": ("A NICU or special newborn care unit", "__nicu__"),
    "ventilation": ("Newborn ventilation", "sig_ventilation"),
    "phototherapy": ("Newborn phototherapy", "sig_phototherapy"),
    "warmer": ("Warmer or incubator", "sig_warmer"),
    "picu": ("Paediatric intensive care", "sig_picu"),
    "ambulance": ("Ambulance", "sig_ambulance"),
    "mfm": ("Maternal-fetal medicine", "sig_mfm"),
    "normal_delivery": ("Normal delivery", "sig_normal_delivery"),
    "painless": ("Pain relief in labour (epidural)", "sig_painless"),
    "antenatal": ("Checkups during pregnancy", "sig_antenatal"),
    "postnatal": ("Care after the birth", "sig_postnatal"),
    "vaccination": ("Vaccination", "sig_vaccination"),
    "ultrasound": ("Ultrasound", "sig_ultrasound"),
    "lab": ("Laboratory tests", "sig_lab"),
}
# Each answer: (id, what the user ticks, what the record must mention, what is also shown,
# a reminder for the phone call). The link from a sign to what to look for was written by
# the team and has not been checked against a clinical source: it is a way to search
# records, not a diagnosis.
QUESTIONS = {
    "high_risk_delivery": [
        ("Is any of this happening?", [
            ("bleeding", "Heavy bleeding now", ["blood", "emergency"], [], ""),
            ("pressure", "Very high blood pressure, fits or a severe headache",
             ["emergency"], ["mfm"], ""),
            ("waters", "Her waters broke before 37 weeks", ["nicu"], [], ""),
            ("fever", "Fever after delivery", ["emergency"], [], ""),
        ]),
        ("About this pregnancy", [
            ("highrisk", "A doctor has said the pregnancy is high-risk", ["csection"], ["mfm"], ""),
            ("early", "She is less than 37 weeks pregnant", ["nicu"], [], ""),
            ("previous", "She has had a C-section or surgery on the womb before",
             ["csection"], [], ""),
            ("twins", "Twins or more", ["nicu", "csection"], [], ""),
        ]),
        ("How would you like to give birth?", [
            ("pref_normal", "Hoping for a normal delivery", [], ["normal_delivery"], ""),
            ("pref_painless", "Wants pain relief in labour (epidural)", [], ["painless"], ""),
            ("pref_csection", "A C-section is planned or likely", [], ["csection"], ""),
        ]),
        ("What else do you want in the same place?", [
            ("same_place", "Newborn care", ["nicu"], [], ""),
            ("want_antenatal", "Checkups during pregnancy", [], ["antenatal"], ""),
            ("want_postnatal", "Care after the birth", [], ["postnatal"], ""),
            ("want_vaccination", "Vaccination for the baby", [], ["vaccination"], ""),
            ("want_ultrasound", "Ultrasound scans", [], ["ultrasound"], ""),
            ("want_lab", "Blood tests", [], ["lab"], ""),
        ]),
        ("Getting there", [
            ("transport", "She needs an ambulance to get there", [], ["ambulance"], ""),
        ]),
    ],
    "nicu": [
        ("Is any of this happening?", [
            ("breathing", "Breathing difficulty or turning blue", ["nicu", "ventilation"], [], ""),
            ("yellow", "Yellow skin or eyes", ["phototherapy"], [], ""),
            ("small", "Born early or very small", ["nicu"], ["warmer"], ""),
            ("sleepy", "Fever, not feeding or unusually sleepy", ["nicu", "emergency"], [], ""),
            ("fits", "Fits (seizures)", ["picu"], [], ""),
        ]),
        ("What is expected?", [
            ("expected_early", "The baby is expected before 37 weeks, or very small",
             ["nicu"], ["warmer"], ""),
            ("expected_breathing", "A doctor has said the baby may need help with breathing",
             ["nicu", "ventilation"], [], ""),
            ("expected_twins", "Twins or more are expected", [], [],
             "Two babies may need two places. The records do not give bed numbers, so ask "
             "whether they have room for both."),
            ("expected_transfer", "The baby may need to be moved by ambulance after birth",
             [], ["ambulance"], ""),
            ("expected_vaccination", "Vaccination for the baby in the same place",
             [], ["vaccination"], ""),
        ]),
        ("About the baby", [
            ("twins", "Twins or more need care", [], [],
             "Two babies may need two places. The records do not give bed numbers, so ask "
             "whether they have room for both."),
        ]),
        ("Getting there", [
            ("transport", "The baby needs an ambulance to get there", [], ["ambulance"], ""),
        ]),
    ],
}

# What each situation changes. "needs": what the record must mention for a facility to be
# the first choice or the backup; "wants": shown on each box, not required; "ask": which of
# the answers above are offered (None = all). Like the answers, this link was written by the
# team and has not been checked against a clinical source.
SITUATION_RULES = {
    "Labour has started": {
        "needs": [], "wants": [], "ask": {"waters", "early", "previous", "twins", "transport"},
        "why": ""},
    "The mother has an emergency": {
        "needs": ["emergency"], "wants": [],
        "ask": {"bleeding", "pressure", "fever", "transport"},
        "why": "Because this is an emergency, we look for records that mention 24-hour "
               "emergency care."},
    "Planning where to give birth": {
        "needs": [], "wants": [],
        "ask": {"highrisk", "early", "previous", "twins", "same_place",
                "pref_normal", "pref_painless", "pref_csection", "want_antenatal",
                "want_postnatal", "want_vaccination", "want_ultrasound", "want_lab"},
        "why": ""},
    "The newborn needs care": {
        "needs": [], "wants": [],
        "ask": {"breathing", "yellow", "small", "sleepy", "fits", "twins", "transport"},
        "why": ""},
    "Newborn care, planned ahead": {
        "needs": [], "wants": [],
        "ask": {"expected_early", "expected_breathing", "expected_twins", "expected_transfer",
                "expected_vaccination"},
        "why": "The baby is not born yet, so these questions are about what is expected, "
               "not about what is happening now."},
}


def situation_rule(service: str) -> dict:
    """The rule for the situation chosen at step 2; an empty rule when there is none."""
    rule = SITUATION_RULES.get(st.session_state.get("situation"))
    newborn_rule = st.session_state.get("situation") in ("The newborn needs care",
                                                         "Newborn care, planned ahead")
    if rule and newborn_rule == (service == "nicu"):       # the rule belongs to this patient
        return rule
    # No situation (a home page example, or the patient was changed in the side panel):
    # offer the questions about what is happening now.
    if service == "nicu":
        ask = SITUATION_RULES["The newborn needs care"]["ask"]
    elif st.session_state.get("role", "urgent") == "urgent":
        ask = (SITUATION_RULES["Labour has started"]["ask"]
               | SITUATION_RULES["The mother has an emergency"]["ask"])
    else:
        ask = SITUATION_RULES["Planning where to give birth"]["ask"]
    return {"needs": [], "wants": [], "ask": ask, "why": ""}


def offered(service: str) -> list:
    """The headings and answers offered on the questions page for this situation."""
    ask = situation_rule(service)["ask"]
    out = []
    for heading, options in QUESTIONS.get(service, []):
        kept = [o for o in options if ask is None or o[0] in ask]
        if kept:
            out.append((heading, kept))
    return out


FIELD_LABELS = {
    "capability": "Services listed",
    "procedure": "Procedures listed",
    "equipment": "Equipment listed",
    "description": "Description",
}

LOCATION_NOTES = {
    "original": "Location confirmed: the map position and postal code in the record agree.",
    "original_unchecked": "Location not cross-checked: the record has no valid postal code.",
    "conflict_pin_confirmed": (
        "The map position in the record looked wrong, so the postal code area is used instead. "
        "It matches other facilities in the same city."
    ),
    "conflict_coords_confirmed": (
        "The postal code in the record looked wrong, so the map position is used instead. "
        "It matches other facilities in the same city."
    ),
    "pin_approx": "Approximate location: based on the postal code area only.",
}

MODES = {
    "urgent": "I need care for a patient now",
    "compare": "Compare options or plan a referral",
    "area": "See coverage in an area",
}
VIEW_LABELS = {"urgent": "Suggested plan", "compare": "Full list to compare",
               "area": "Coverage map"}

FAR_BACKUP_KM = 50       # from here on the summary says the backup is far (our choice)
URGENT_RADIUS_KM = 300  # how far the urgent view looks before giving up

GENERIC_NAME_WORDS = {
    "hospital", "hospitals", "centre", "center", "clinic", "nursing", "home", "institute",
    "medical", "college", "care", "health", "healthcare", "multispeciality", "speciality",
    "super", "research", "general", "private", "limited", "the", "and",
}

cfg = Config()
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
HTTP_PATH = f"/sql/1.0/warehouses/{WAREHOUSE_ID}"


# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------

BLOCK = 'div[data-testid="stVerticalBlock"]'


def _scope(name: str, suffix: str = "") -> str:
    """CSS selector for the st.container() that holds marker(name)."""
    return (f'{BLOCK}:has(> .element-container span.mk-{name}){suffix}, '
            f'{BLOCK}:has(> [data-testid="stElementContainer"] span.mk-{name}){suffix}')


def _within(name: str, inner: str) -> str:
    return ", ".join(f"{part.strip()} {inner}" for part in _scope(name).split(", "))


# The taxi icon on the "Show to driver" button (96 px PNG, transparent outside the disc).
TAXI_ICON_BASE64 = (
    "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"
)

# Streamlit gives every element a fixed pixel width. Inside a padded block that width is
# too wide, so make the elements follow their container instead.
_INNER = ('> .element-container', '> [data-testid="stElementContainer"]',
          '> .element-container > div', '> [data-testid="stElementContainer"] > div')
_PADDED_INNER = ", ".join(_within(name, inner)
                          for name in ("band", "planbox", "cream", "side", "card", "soon", "prose")
                          for inner in _INNER)
_FAC_INNER = ", ".join(_within("fac", inner) for inner in _INNER)
_FAB_INNER = ", ".join(_within("fab", inner) for inner in _INNER)
_VIEWTABS_KIDS = ", ".join(_within("viewtabs", inner) for inner in _INNER)
_KIDS = ("> .element-container", '> [data-testid="stElementContainer"]')


def _fac(position: str, inner: str = "") -> str:
    """Selector for one direct element of a facility box, optionally something inside it."""
    return ", ".join(_within("fac", f"{kid}{position}{inner}") for kid in _KIDS)


_FAC_LAST = _fac(":last-child")
_FAC_LAST_DIV = _fac(":last-child", " > div")
_FAC_LAST_BTN = _fac(":last-child", " button")
_FAC_LAST_BTN_BEFORE = _fac(":last-child", " button::before")
_FAC_LAST_BTN_AFTER = _fac(":last-child", " button::after")
_FAC_LAST_BTN_P = _fac(":last-child", " button p")
_FAC_LAST_BTN_P_AFTER = _fac(":last-child", " button p::after")
_FAC_LAST_BTN_HOVER = _fac(":last-child", " button:hover")
_FAC_DETAILS = _fac(":nth-last-child(2)")
_NAVROW_KIDS = ", ".join(_within("navrow", inner) for inner in _INNER)
_QUIET_KIDS = ", ".join(_within("quiet", inner) for inner in _INNER)
_CARD_LAST = ", ".join(_within(name, inner) for name in ("card", "soon")
                       for inner in ("> .element-container:last-child",
                                     '> [data-testid="stElementContainer"]:last-child'))
_BAND_BTN = _within("band", 'button[kind="primary"]')
_BAND_BTN_P = _within("band", 'button[kind="primary"] p')

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Bricolage+Grotesque:wght@500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@700&family=Noto+Sans+Tamil:wght@700&family=Noto+Sans+Gujarati:wght@700&family=Noto+Sans+Telugu:wght@700&family=Noto+Sans+Bengali:wght@700&family=Noto+Sans+Kannada:wght@700&family=Noto+Sans+Gurmukhi:wght@700&family=Noto+Sans+Malayalam:wght@700&display=swap');
.stApp {{ background: #FFFFFF; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th,
.stApp div[data-baseweb="select"] {{ font-family: 'Atkinson Hyperlegible', sans-serif; }}
.block-container {{ max-width: 1160px; padding-top: 2.6rem; padding-bottom: 5rem; }}
/* Streamlit's header bar is see-through but lies over the top 60 px of the page, where it
   swallowed clicks meant for our own top-right buttons. Let clicks pass through it, except
   on its own controls. */
header[data-testid="stHeader"] {{ background: transparent; pointer-events: none; }}
header[data-testid="stHeader"] [data-testid="stToolbar"], header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a {{ pointer-events: auto; }}
.stApp h1, .stApp h2 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; }}
.stApp a {{ color: {PLUM}; }}
.stApp a:hover {{ color: {PLUM_DARK}; }}
div.element-container:has(span.mk), div[data-testid="stElementContainer"]:has(span.mk) {{ display: none; }}
.stApp [data-testid="stHeaderActionElements"] {{ display: none; }}

/* buttons */
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {{ background: {PLUM};
  border: 0; color: #FFFFFF; border-radius: 4px; min-height: 48px; padding: 0 28px; }}
.stApp button[kind="primary"] p, .stApp button[kind="primaryFormSubmit"] p {{ color: #FFFFFF;
  font-weight: 700; font-size: 17px; }}
.stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover {{
  background: {PLUM_DARK}; color: #FFFFFF; }}
.stApp button[kind="secondary"] {{ background: #FFFFFF; border: 1.5px solid {PLUM}; border-radius: 4px;
  min-height: 46px; padding: 0 20px; }}
.stApp button[kind="secondary"] p {{ color: {PLUM}; font-weight: 700; font-size: 16px; }}
.stApp button[kind="secondary"]:hover {{ border-color: {PLUM_DARK}; background: #FBF6FA; }}
.stApp button:disabled, .stApp button:disabled:hover {{ background: #F3E7C8; border: 1.5px dashed #9A8F7A; }}
.stApp button:disabled p {{ color: #5A5142; font-weight: 400; }}

/* top bar: links sit in one row at the right edge */
{_scope('navrow')} {{ flex-direction: row; justify-content: flex-end; align-items: center;
  flex-wrap: wrap; gap: 8px 10px; }}
{_NAVROW_KIDS} {{ width: auto !important; }}
{_within('navrow', 'button')} {{ background: #FFFFFF; border: 1.5px solid {PLUM}; border-radius: 6px;
  min-height: 40px; padding: 0 14px; }}
{_within('navrow', 'button p')} {{ color: {PLUM}; font-weight: 700; font-size: 15px; text-decoration: none; }}
{_within('navrow', 'button:hover')} {{ background: {CREAM}; border-color: {PLUM_DARK}; }}
{_within('toright', '.stButton')} {{ display: flex; justify-content: flex-end; }}
{_scope('prose')} {{ max-width: 780px; }}
{_scope('card')}, {_scope('soon')} {{ display: flex; flex-direction: column; }}
{_CARD_LAST} {{ margin-top: auto; }}

/* home page: fills exactly one screen, whatever its height. The sections share the spare
   height equally (space-between), and the big pieces grow a little on taller screens. */
.block-container:has(span.mk-home) {{ padding-top: 1.25rem; padding-bottom: 0.75rem; }}
{_scope('home')} {{ min-height: calc(100vh - 2.25rem); justify-content: space-between; }}
{_within('home', '> div')} {{ flex-grow: 0 !important; }}
{_scope('tight')} {{ gap: 0.6rem; }}
{_within('hero', 'img')} {{ height: clamp(140px, calc(31vh - 36px), 340px); object-fit: cover;
  object-position: center 30%; }}
{_within('sec', '.sa .sa-h1')} {{ font-size: clamp(28px, 4.9vh, 54px); text-wrap: balance; }}
{_within('sec', '.sa .sa-lede')} {{ margin-top: 12px; font-size: clamp(16px, 2.3vh, 20px); max-width: 600px; }}
{_within('sec', '.sa .sa-h2.big')} {{ font-size: clamp(24px, 3.6vh, 32px); }}
{_within('sec', '.sa .sa-h3')} {{ font-size: clamp(18px, 2.3vh, 22px); }}
{_within('sec', '.sa .sa-small')} {{ font-size: clamp(16px, 1.9vh, 18px); }}
{_within('sec', '.sa .sa-h2.sub')} {{ font-size: clamp(22px, 2.8vh, 26px); }}
{_within('sec', '.sa .sa-fine')} {{ font-size: clamp(14px, 1.6vh, 16px); }}
{_within('sec', 'button[kind="secondary"]')} {{ min-height: clamp(40px, 5vh, 52px); }}

/* text links that are really buttons */
{_within('links', 'button')} {{ background: transparent; border: 0; min-height: 44px; padding: 0; }}
{_within('links', 'button p')} {{ color: {PLUM}; font-weight: 400; font-size: 16px; text-decoration: underline; }}
{_within('links', 'button:hover')} {{ background: transparent; }}
{_within('boldlinks', 'button')} {{ background: transparent; border: 0; min-height: 32px !important; padding: 0; }}
{_within('boldlinks', 'button p')} {{ color: {PLUM}; font-weight: 700; font-size: 17px; text-decoration: underline; }}
{_within('boldlinks', 'button:hover')} {{ background: transparent; }}

/* blocks */
{_scope('band')}, {_scope('rule')}, {_scope('cream')}, {_scope('side')}, {_scope('card')},
{_scope('soon')}, {_scope('hero')}, {_scope('fac')} {{ box-sizing: border-box; }}
{_scope('band')} {{ background: {RED}; border-radius: 4px; padding: clamp(16px, 3.6vh, 40px) 36px; }}
{_PADDED_INNER} {{ width: 100% !important; max-width: 100% !important; }}
{_BAND_BTN} {{ background: #FFFFFF; min-height: 50px; padding: 0 40px; }}
{_BAND_BTN_P} {{ color: #8C1D18; font-size: 19px; }}
{_scope('rule')} {{ border-top: 3px solid {YELLOW}; padding-top: clamp(10px, 2.2vh, 22px); }}
/* the second entry on the home page: a box like the red one, in a calm colour */
{_scope('planbox')} {{ box-sizing: border-box; background: {CREAM}; border-left: 6px solid {YELLOW};
  border-radius: 4px; padding: clamp(14px, 3vh, 34px) 36px; }}
{_within('planbox', 'button[kind="primary"]')} {{ min-height: 50px; padding: 0 32px; }}
/* examples: one quiet line of links */
{_scope('quiet')} {{ flex-direction: row; flex-wrap: wrap; align-items: center; gap: 2px 20px; }}
{_QUIET_KIDS} {{ width: auto !important; }}
.stApp .sa .sa-quiet {{ margin: 0; font-size: 15px; color: {MUTED}; }}
{_within('quiet', 'button')} {{ background: transparent; border: 0; min-height: 36px !important; padding: 0; }}
{_within('quiet', 'button p')} {{ color: {MUTED}; font-weight: 400; font-size: 15px;
  text-decoration: underline; text-decoration-color: #C9BFAE; text-underline-offset: 3px; }}
{_within('quiet', 'button:hover')} {{ background: transparent; }}
{_within('quiet', 'button:hover p')} {{ color: {PLUM}; text-decoration-color: {PLUM}; }}
{_scope('cream')} {{ background: {CREAM}; border-radius: 4px; padding: 40px; }}
{_scope('side')} {{ background: {CREAM}; border-radius: 4px; padding: 28px; }}
{_scope('card')} {{ background: #FFFFFF; border-radius: 4px; border: 1.5px solid #FFFFFF;
  border-bottom: 4px solid {PLUM}; padding: 22px; min-height: 212px; }}
{_scope('soon')} {{ background: #FBF4E1; border-radius: 4px; border: 1.5px dashed #9A8F7A;
  padding: 22px; min-height: 188px; }}
{_scope('hero')} {{ background: {CREAM}; border-radius: 4px; overflow: hidden; gap: 0; }}
/* a facility: a box that opens its details when clicked anywhere. The button inside it is
   the real control; its ::after is stretched over the whole box, so keyboard and screen
   reader users get an ordinary button and mouse users get a large target. */
{_scope('fac')} {{ position: relative; background: #FFFFFF; border: 1.5px solid {RULE};
  border-radius: 6px; padding: 20px 24px 12px 24px; margin-bottom: 6px; gap: 0.4rem;
  transition: border-color .12s, box-shadow .12s; }}
{_scope('fac', ':hover')} {{ border-color: {PLUM}; box-shadow: 0 2px 10px rgba(91, 42, 78, .14); }}
{_scope('fac', ':has(button:focus-visible)')} {{ outline: 3px solid {YELLOW}; outline-offset: 2px; }}
{_FAC_INNER} {{ width: 100% !important; max-width: 100% !important; position: static !important; }}
{_within('fac', '.stButton')}, {_within('fac', '[data-testid="stButton"]')} {{ position: static !important; }}
{_within('fac', 'button')} {{ position: static !important; background: transparent; border: 0;
  min-height: 40px; padding: 0; outline: none !important; box-shadow: none !important; }}
{_within('fac', 'button::after')} {{ content: ""; position: absolute; inset: 0; border-radius: 6px;
  cursor: pointer; }}
{_within('fac', 'button p')} {{ color: {PLUM}; font-weight: 700; font-size: 16px;
  text-decoration: underline; text-align: left; }}
{_within('fac', 'button p::after')} {{ content: " \\203A"; }}
{_within('fac', 'button:hover')}, {_within('fac', 'button:active')}, {_within('fac', 'button:focus')}
  {{ background: transparent; color: {PLUM}; }}
/* the driver button: the last element of the box, pinned to its bottom right corner and
   lifted above the stretched details button */
{_FAC_LAST} {{ position: absolute !important; right: 18px; bottom: 12px; width: auto !important;
  max-width: none !important; z-index: 3; }}
{_FAC_LAST_DIV} {{ width: auto !important; }}
{_FAC_LAST_BTN} {{ position: relative !important; display: inline-flex; align-items: center;
  gap: 8px; min-height: 44px; padding: 0 14px 0 4px; background: #FFFFFF;
  border: 1.5px solid {PLUM}; border-radius: 22px; }}
{_FAC_LAST_BTN_BEFORE} {{ content: ""; flex: 0 0 auto; width: 36px; height: 36px;
  background: url("data:image/png;base64,{TAXI_ICON_BASE64}") center / contain no-repeat; }}
{_FAC_LAST_BTN_AFTER} {{ content: none; }}
{_FAC_LAST_BTN_P} {{ text-decoration: none; font-size: 15px; }}
{_FAC_LAST_BTN_P_AFTER} {{ content: none; }}
{_FAC_LAST_BTN_HOVER} {{ background: {CREAM}; }}
/* keep the details link clear of the driver button */
{_FAC_DETAILS} {{ padding-right: 176px; box-sizing: border-box; }}
@media (max-width: 640px) {{
  {_FAC_LAST_BTN} {{ padding: 0 4px; }}
  /* icon only on a phone, but the words stay there for screen readers */
  {_FAC_LAST_BTN_P} {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip-path: inset(50%); white-space: nowrap; }}
  {_FAC_DETAILS} {{ padding-right: 60px; }}
}}
{_scope('facplain')} {{ border-bottom: 1px solid {RULE}; padding-bottom: 24px; margin-bottom: 12px; }}
/* the card for the driver */
.stApp .sa .sa-driver .say {{ font-size: 30px; line-height: 1.4; font-weight: 700; color: {PLUM_DARK};
  font-family: 'Noto Sans Devanagari', 'Noto Sans Tamil', 'Noto Sans Gujarati', 'Noto Sans Telugu',
  'Noto Sans Bengali', 'Noto Sans Kannada', 'Noto Sans Gurmukhi', 'Noto Sans Malayalam',
  'Atkinson Hyperlegible', sans-serif; }}
.stApp .sa .sa-driver .en {{ margin-top: 4px; font-size: 16px; color: {MUTED}; }}
.stApp .sa .sa-driver .name {{ margin-top: 18px; font-size: 30px; line-height: 1.25; font-weight: 700;
  color: {INK}; }}
.stApp .sa .sa-driver .addr {{ margin-top: 6px; font-size: 22px; line-height: 1.4; color: {INK}; }}
.stApp .sa .sa-driver .tel {{ margin-top: 12px; font-size: 22px; font-weight: 700; color: {INK}; }}
.stApp .sa .sa-driver-map {{ margin-top: 14px; font-size: 16px; line-height: 1.5; color: {MUTED}; }}
.stApp .sa .sa-driver-map a {{ display: inline-block; margin-right: 8px; padding: 10px 18px;
  background: {PLUM}; color: #FFFFFF !important; font-size: 17px; font-weight: 700;
  border-radius: 4px; text-decoration: none; }}
.stApp .sa ul.sa-driver-numbers {{ list-style: none; margin: 6px 0 10px 0; padding: 0; }}
.stApp .sa ul.sa-driver-numbers li {{ margin: 0 0 6px 0; font-size: 17px; color: {INK}; }}
.stApp .sa ul.sa-driver-numbers a {{ display: inline-block; min-width: 52px; font-size: 20px;
  font-weight: 700; }}
/* the "Ask Genie" button: pinned to the bottom right corner of the window */
{_scope('fab')} {{ position: fixed; right: 28px; bottom: 28px; width: auto !important; z-index: 999; }}
{_FAB_INNER} {{ width: auto !important; }}
/* its empty wrapper must not count as a section of the home page */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > .element-container span.mk-fab),
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] span.mk-fab),
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] span.mk-fab),
div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] span.mk-fab)
  {{ position: absolute; }}
{_within('fab', 'button')} {{ display: inline-flex; align-items: center; gap: 10px; min-height: 52px;
  padding: 0 22px 0 18px; background: {PLUM}; border: 2px solid #FFFFFF; border-radius: 26px;
  box-shadow: 0 4px 14px rgba(43, 35, 48, .28); }}
{_within('fab', 'button::before')} {{ content: ""; flex: 0 0 auto; width: 24px; height: 24px;
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 5h16v11H9l-5 4z'/><path d='M8 9h8M8 12.5h5'/></svg>") center / contain no-repeat; }}
{_within('fab', 'button p')} {{ color: #FFFFFF; font-weight: 700; font-size: 16px; }}
{_within('fab', 'button:hover')} {{ background: {PLUM_DARK}; border-color: #FFFFFF; }}
@media (max-width: 640px) {{
  {_scope('fab')} {{ right: 14px; bottom: 14px; }}
  {_within('fab', 'button')} {{ padding: 0 14px; }}
  {_within('fab', 'button p')} {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip-path: inset(50%); white-space: nowrap; }}
}}
/* the list / map switch on the planning views */
{_scope('viewtabs')} {{ flex-direction: row; gap: 0; margin: 4px 0 6px 0; }}
{_VIEWTABS_KIDS} {{ width: auto !important; }}
{_within('viewtabs', 'button')} {{ min-height: 42px; padding: 0 22px; border: 1.5px solid {PLUM};
  border-radius: 0; background: #FFFFFF; }}
{_within('viewtabs', 'button p')} {{ color: {PLUM}; font-weight: 700; font-size: 16px; }}
{_within('viewtabs', 'button[kind="primary"]')} {{ background: {PLUM}; }}
{_within('viewtabs', 'button[kind="primary"] p')} {{ color: #FFFFFF; }}
/* links inside the box stay clickable above the stretched button */
.stApp .sa .sa-fac-phone a, .stApp .sa .sa-fac-sum a, .stApp .sa .sa-callbox a {{ position: relative; z-index: 2; }}
.stApp .sa .sa-callbox {{ margin: 4px 0 14px 0; padding: 14px 16px; background: {CREAM}; border-radius: 4px; }}
.stApp .sa .sa-callbox a.sa-call {{ display: inline-block; background: {RED}; color: #FFFFFF; font-weight: 700;
  font-size: 19px; text-decoration: none; padding: 12px 26px; border-radius: 4px; }}
.stApp .sa .sa-callbox a.sa-call:hover {{ color: #FFFFFF; filter: brightness(0.92); }}
.stApp .sa .sa-callbox p {{ margin: 12px 0 4px 0; font-size: 16px; font-weight: 700; color: {INK}; }}
.stApp .sa .sa-callbox ol, .stApp .sa ol.sa-askq {{ margin: 0 0 0 20px; padding: 0; font-size: 16px; line-height: 1.5; color: {INK}; }}
.stApp .sa a.sa-i {{ display: inline-flex; align-items: center; justify-content: center; width: 17px;
  height: 17px; margin: 0 1px 0 4px; border: 1.5px solid {PLUM}; border-radius: 50%; color: {PLUM};
  font-family: Georgia, 'Times New Roman', serif; font-size: 12px; font-weight: 700; font-style: italic;
  line-height: 1; text-decoration: none; vertical-align: 2px; position: relative; z-index: 2; }}
.stApp .sa a.sa-i:hover, .stApp .sa a.sa-i:focus {{ background: {PLUM}; color: #FFFFFF; }}
.stApp .sa .sa-share {{ margin-top: 12px; font-size: 16px; line-height: 1.6; color: {MUTED}; }}
.stApp .sa .sa-share a {{ display: inline-block; margin: 4px 8px 4px 0; padding: 8px 16px; border: 1.5px solid {PLUM};
  border-radius: 4px; font-weight: 700; text-decoration: none; }}
.stApp .sa .sa-fac-sum {{ margin-top: 2px; font-size: 17px; line-height: 1.5; color: {INK}; }}

/* on a phone the search panel comes before the results, not after them */
@media (max-width: 640px) {{
  div[data-testid="stHorizontalBlock"]:has(span.mk-side) {{ flex-wrap: wrap-reverse; }}
  {_scope('navrow')} {{ justify-content: flex-start; }}
  {_scope('band')}, {_scope('cream')}, {_scope('planbox')} {{ padding: 24px; }}
  {_within('toright', '.stButton')} {{ justify-content: flex-start; }}
  .stApp .sa .sa-h1 {{ font-size: 36px; }}
  .stApp .sa .sa-h1.small {{ font-size: 30px; }}
}}

/* pieces written as HTML: every block is wrapped in <div class="sa">.
   Streamlit pulls the next element up after a text block; ours keep the normal gap. */
.stApp [data-testid="stMarkdownContainer"]:has(> .sa) {{ margin-bottom: 0 !important; }}
.stApp .sa p, .stApp .sa h1, .stApp .sa h2, .stApp .sa h3 {{ margin: 0; padding: 0; }}
.stApp .sa .sa-brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 24px; font-weight: 700;
  line-height: 44px; color: {PLUM}; }}
.stApp .sa .sa-h1 {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 40px; line-height: 1.1;
  font-weight: 700; color: {PLUM}; }}
.stApp .sa .sa-h1.small {{ font-size: 38px; line-height: 1.12; }}
.stApp .sa .sa-lede {{ margin-top: 24px; max-width: 520px; font-size: 20px; line-height: 1.55; color: {MUTED}; }}
.stApp .sa .sa-h2 {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 26px; line-height: 1.2;
  font-weight: 700; color: {INK}; }}
.stApp .sa .sa-h2.big {{ font-size: 28px; line-height: 1.15; }}
.stApp .sa .sa-h2.sub {{ font-size: 22px; }}
.stApp .sa .sa-fine {{ font-size: 14px; line-height: 1.45; color: {MUTED}; max-width: 900px; }}
.stApp .sa .sa-h2.ruled {{ font-size: 22px; margin: 14px 0 8px 0; padding-bottom: 10px;
  border-bottom: 3px solid {YELLOW}; }}
.stApp .sa .sa-h3 {{ font-family: 'Atkinson Hyperlegible', sans-serif; font-size: 18px; line-height: 1.3;
  font-weight: 700; color: {INK}; }}
.stApp .sa .sa-p {{ margin-top: 6px; font-size: 17px; line-height: 1.5; color: {MUTED}; }}
.stApp .sa .sa-p.lg {{ margin-top: 0; font-size: 18px; color: {INK}; }}
.stApp .sa .sa-small {{ font-size: 16px; line-height: 1.5; color: {MUTED}; }}
.stApp .sa .sa-card-title, .stApp .sa .sa-soon-title {{ font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 21px; line-height: 1.25; font-weight: 700; color: {PLUM}; }}
.stApp .sa .sa-soon-title {{ color: #5A5142; }}
.stApp .sa p.sa-on-red, .stApp .sa h2.sa-on-red {{ color: #FFFFFF !important; }}
.stApp .sa .sa-call {{ padding: 20px 24px; border-left: 8px solid {RED}; background: {RED_TINT}; }}
.stApp .sa .sa-call .t {{ font-size: 20px; font-weight: 700; line-height: 1.35; color: {RED_DARK}; }}
.stApp .sa .sa-call .d {{ margin-top: 6px; font-size: 17px; line-height: 1.5; color: {INK}; }}
.stApp .sa .sa-work {{ width: 100%; margin-top: 14px; border-collapse: collapse; font-size: 15px;
  line-height: 1.4; border: 0; }}
.stApp .sa .sa-work caption {{ caption-side: top; padding: 0 0 6px 0; text-align: left; font-weight: 700; color: {INK}; }}
.stApp .sa .sa-work th, .stApp .sa .sa-work td {{ padding: 8px 10px 8px 0; border: 0;
  border-top: 1px solid {RULE}; text-align: left; vertical-align: top; color: {INK}; }}
.stApp .sa .sa-work th {{ font-weight: 700; }}
.stApp .sa .sa-work .nc {{ display: block; font-size: 13px; color: {MUTED}; }}
.stApp .sa .sa-sit-line {{ margin: 8px 0 6px 0; }}
/* cards in one row are the same height, whatever the length of their text */
:is(div[data-testid="column"], div[data-testid="stColumn"]):has(span.mk-card),
:is(div[data-testid="column"], div[data-testid="stColumn"]):has(span.mk-card) > div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > .element-container span.mk-card),
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] span.mk-card),
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] span.mk-card) {{
  display: flex; flex-direction: column; flex: 1 1 auto; }}
.stApp .sa .sa-card-head {{ display: flex; align-items: flex-start; justify-content: space-between;
  gap: 18px; }}
.stApp .sa img.sa-card-icon {{ flex: 0 0 auto; width: 72px; height: 72px; max-width: none; margin: 0; }}
.stApp .sa .sa-path {{ margin-bottom: 18px; padding: 16px 24px; border-radius: 4px; }}
.stApp .sa .sa-path .k {{ font-size: 13px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
.stApp .sa .sa-path .t {{ margin-top: 2px; font-family: 'Bricolage Grotesque', sans-serif; font-size: 24px;
  font-weight: 700; line-height: 1.25; }}
.stApp .sa .sa-path .g {{ margin-top: 4px; font-size: 16px; line-height: 1.45; }}
.stApp .sa .sa-path.urgent {{ background: {RED}; }}
.stApp .sa .sa-path.urgent p {{ color: #FFFFFF !important; }}
.stApp .sa .sa-path.quiet {{ background: #FFFFFF; border: 1.5px solid {CREAM_RULE}; border-top: 4px solid {YELLOW}; }}
.stApp .sa .sa-path.quiet .k {{ color: {MUTED}; }}
.stApp .sa .sa-path.quiet .t {{ color: {INK}; }}
.stApp .sa .sa-path.quiet .g {{ color: {MUTED}; }}
.stApp .sa ul.sa-told {{ margin: 0 0 4px 0; padding-left: 20px; font-size: 16px; line-height: 1.45;
  color: {INK}; }}
.stApp .sa .sa-plan {{ margin-top: 16px; padding: 20px 24px; border-left: 8px solid {PLUM};
  background: {CREAM}; }}
.stApp .sa .sa-plan .t {{ font-size: 20px; font-weight: 700; line-height: 1.35; color: {PLUM_DARK}; }}
.stApp .sa .sa-plan .d {{ margin-top: 8px; font-size: 17px; line-height: 1.5; color: {INK}; }}
.stApp .sa .sa-plan .f {{ margin-top: 12px; font-size: 15px; line-height: 1.5; color: {MUTED}; }}
.stApp .sa .sa-note {{ padding: 14px 18px; border-left: 8px solid {YELLOW}; background: {CREAM};
  font-size: 16px; line-height: 1.5; color: {INK}; }}
.stApp .sa .sa-fac-head {{ display: flex; flex-wrap: wrap; justify-content: space-between;
  align-items: baseline; gap: 4px 12px; }}
.stApp .sa .sa-fac-name {{ font-family: 'Atkinson Hyperlegible', sans-serif; font-size: 26px;
  font-weight: 700; line-height: 1.2; color: {INK}; }}
.stApp .sa .sa-fac-dist, .stApp .sa .sa-fac-addr {{ font-size: 17px; line-height: 1.5; color: {MUTED}; }}
.stApp .sa .sa-fac-addr {{ margin-top: 8px; }}
.stApp .sa .sa-fac-phone {{ margin: 8px 0 10px 0; font-size: 22px; line-height: 1.4; font-weight: 700; color: {INK}; }}
.stApp .sa .sa-table {{ width: 100%; border-collapse: collapse; font-size: 16px; line-height: 1.45; border: 0; }}
.stApp .sa .sa-table caption {{ padding: 0 0 8px 0; text-align: left; font-size: 17px; font-weight: 700;
  color: {INK}; caption-side: top; }}
.stApp .sa .sa-table th, .stApp .sa .sa-table td {{ padding: 10px 12px 10px 0; border: 0;
  border-top: 1px solid {RULE}; text-align: left; vertical-align: top; color: {INK}; }}
.stApp .sa .sa-table tr:last-child th, .stApp .sa .sa-table tr:last-child td {{ border-bottom: 1px solid {RULE}; }}
.stApp .sa .sa-table th {{ width: 26%; font-weight: 700; }}
.stApp .sa .sa-table td.s {{ width: 24%; font-weight: 700; }}
.stApp .sa .sa-table td.s.no, .stApp .sa .sa-table td.d.no {{ font-weight: 400; font-style: italic; color: {MUTED}; }}
.stApp .sa .sa-table td.d {{ color: {MUTED}; padding-right: 0; }}
.stApp .sa .sa-table .nc {{ display: block; font-weight: 400; font-size: 14px; color: {MUTED}; }}
.stApp .sa .sa-flag {{ font-style: italic; color: {RED_DARK}; }}
.stApp .sa .sa-side-h {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 20px; line-height: 1.3;
  font-weight: 700; color: {INK}; }}
.stApp .sa .sa-side-rule {{ border: 0; border-top: 1px solid {CREAM_RULE}; margin: 4px 0 18px 0; }}
.stApp .sa ul {{ margin: 6px 0 10px 0; }}
.stApp .sa a {{ overflow-wrap: anywhere; }}
.stApp .sa .sa-step {{ margin-bottom: 6px; font-size: 15px; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: {MUTED}; }}
.stApp [data-testid="stForm"] {{ border: 0; padding: 0; }}
.stApp [data-testid="stForm"] div[data-baseweb="input"],
.stApp [data-testid="stForm"] div[data-baseweb="base-input"] {{ height: 54px; }}
.stApp [data-testid="stForm"] input {{ font-size: 19px; }}
.stApp [data-testid="stForm"] [data-testid="InputInstructions"] {{ display: none; }}
.stApp [data-testid="stForm"] label p {{ font-size: 17px; font-weight: 700; color: {INK}; }}
</style>
"""


def marker(name: str) -> None:
    """Tag the surrounding st.container() so the CSS above can style it."""
    st.markdown(f'<span class="mk mk-{name}"></span>', unsafe_allow_html=True)


def esc(value) -> str:
    """Make data text safe inside HTML ($ would start a formula in Streamlit)."""
    return html.escape(str(value)).replace("$", "&#36;")


def space(px: int) -> None:
    """Extra air between two sections (on top of the normal gap between elements)."""
    st.markdown(f'<div class="sa" style="height:{int(px)}px"></div>', unsafe_allow_html=True)


def show_html(markup: str, where=None) -> None:
    """Write a block of our own HTML, here or in the given column or container."""
    (where or st).markdown(f'<div class="sa">{markup}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# "Use my current location": a small button that asks the browser where the user is.
# Its page is written to a temporary folder when the app starts, so the app stays one file.
# If that fails for any reason the button is simply not shown.
# ----------------------------------------------------------------------------

GEO_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap">
<style>
body { margin: 0; font-family: 'Atkinson Hyperlegible', sans-serif; background: transparent; }
button { min-height: 48px; padding: 0 20px; border: 1.5px solid #5B2A4E; border-radius: 4px;
  background: #FFFFFF; color: #5B2A4E; font: 700 16px 'Atkinson Hyperlegible', sans-serif; cursor: pointer; }
button:hover { background: #FBF6FA; }
button:disabled { opacity: 0.6; cursor: default; }
p { margin: 8px 0 0 0; font-size: 15px; color: #4E4554; }
p:empty { display: none; }
</style></head><body>
<button id="go" type="button">Use my current location</button>
<p id="note" aria-live="polite"></p>
<script>
function send(type, data) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
}
function resize() { send("streamlit:setFrameHeight", { height: document.body.scrollHeight + 6 }); }
function answer(value) { send("streamlit:setComponentValue", { value: value, dataType: "json" }); }
window.addEventListener("message", function (event) {
  if (event.data && event.data.type === "streamlit:render") { resize(); }
});
var button = document.getElementById("go");
var note = document.getElementById("note");
button.addEventListener("click", function () {
  if (!navigator.geolocation) {
    note.textContent = "This browser cannot share a location.";
    answer({ error: "unsupported", at: Date.now() });
    resize();
    return;
  }
  button.disabled = true;
  note.textContent = "Waiting for your browser...";
  resize();
  navigator.geolocation.getCurrentPosition(function (position) {
    note.textContent = "Location received.";
    answer({ lat: position.coords.latitude, lon: position.coords.longitude,
             accuracy: position.coords.accuracy, at: Date.now() });
  }, function (problem) {
    button.disabled = false;
    note.textContent = "";
    answer({ error: String(problem && problem.code), at: Date.now() });
    resize();
  }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 });
});
send("streamlit:componentReady", { apiVersion: 1 });
resize();
// On a desktop, put the cursor in the text box so the user can type straight away.
try {
  var page = window.parent;
  if (page.innerWidth > 640) {
    var box = page.document.querySelector('[data-testid="stForm"] input[type="text"]');
    if (box && !box.value) { box.focus(); }
  }
} catch (ignored) {}
</script></body></html>
"""


def make_geo_button():
    try:
        import tempfile
        import streamlit.components.v1 as components
        folder = os.path.join(tempfile.gettempdir(), "safearrival_geo_button_v2")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as page:
            page.write(GEO_PAGE)
        return components.declare_component("safearrival_geo_button", path=folder)
    except Exception:
        return None


GEO_BUTTON = make_geo_button()


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------

def run_query(query: str) -> pd.DataFrame:
    """Run one SQL statement on the SQL warehouse as the app's service principal."""
    host = cfg.host.replace("https://", "").replace("http://", "")
    with sql.connect(
        server_hostname=host,
        http_path=HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


EXTRA_V4 = ["sig_ventilation", "sig_phototherapy", "sig_warmer", "sig_picu",
            "sig_ambulance", "sig_mfm"]
EXTRA_V5 = ["sig_normal_delivery", "sig_painless", "sig_antenatal", "sig_postnatal",
            "sig_vaccination", "sig_ultrasound", "sig_lab"]
EXTRA_COLUMNS = EXTRA_V4 + EXTRA_V5


@st.cache_data(ttl=600, show_spinner=False)
def get_recommendations(service: str, lat: float, lon: float, radius_km: float,
                        sort: str, limit: int) -> pd.DataFrame:
    # service and sort come from fixed lists; the rest are numbers.
    def query(extra: str) -> str:
        return f"""
        SELECT result_rank, list_section, unique_id, name, address_line1, city, state, pin,
               phone, website, n_phone_numbers, evidence_tier, distance_km,
               distance_km_by_coords, location_quality, lat, lon, n_sources,
               hours_level, hours_quote,
               sig_csection, sig_blood, sig_newborn, sig_surgery, sig_highrisk,
               nicu_level, nicu_quote,{extra}
               array_join(evidence_quotes, '{SEP}')          AS quotes,
               array_join(slice(source_urls, 1, 5), '{SEP}') AS sources
        FROM {SCHEMA}.recommend_at('{service}', {float(lat)}, {float(lon)},
                                   CAST({float(radius_km)} AS DOUBLE), '{sort}', {int(limit)})
        ORDER BY result_rank
        """
    # The database may be on an older version of the SQL. Ask for the newest columns first
    # and fall back one version at a time; a signal the database does not have reads as
    # "not mentioned". Everything else works the same.
    for columns in (EXTRA_COLUMNS, EXTRA_V4, []):
        try:
            df = run_query(query(" " + ", ".join(columns) + "," if columns else ""))
        except Exception:
            if not columns:
                raise
            continue
        for column in EXTRA_COLUMNS:
            if column not in columns:
                df[column] = None
        df["extras_ok"] = bool(columns)
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def facility_size(unique_id: str):
    """(beds, doctors) as the record gives them; None for a number that is missing or was
    flagged as implausible when the data was cleaned. Shown to planners, never used to rank."""
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", str(unique_id)):
        return None, None
    try:
        df = run_query(f"""
            SELECT CASE WHEN beds_implausible THEN NULL ELSE n_beds END       AS beds,
                   CASE WHEN doctors_implausible THEN NULL ELSE n_doctors END AS doctors
            FROM {SCHEMA}.facilities_final WHERE unique_id = '{unique_id}' LIMIT 1
        """)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    row = df.iloc[0]
    return tuple(int(v) if has(v) and float(v) > 0 else None for v in (row["beds"], row["doctors"]))


# The dataset's own facility table: the only place that says who runs a facility.
RAW_FACILITIES = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
PUBLIC_WORDS = ("public", "government")


@st.cache_data(ttl=3600, show_spinner=False)
def operator_types(unique_ids: tuple):
    """{unique_id: 'private' | 'public' | ...} from the dataset's operatorTypeId column.
    None when the table cannot be read (for example the app has no permission on it)."""
    ids = [i for i in unique_ids if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", str(i))]
    if not ids:
        return {}
    quoted = ", ".join(f"'{i}'" for i in ids)
    try:
        df = run_query(f"SELECT unique_id, operatorTypeId FROM {RAW_FACILITIES} "
                       f"WHERE unique_id IN ({quoted})")
    except Exception:
        return None
    return {row["unique_id"]: str(row["operatorTypeId"]).strip().lower()
            for _, row in df.iterrows() if has(row["operatorTypeId"])}


def operator_html(row) -> str:
    types = operator_types((row["unique_id"],))
    kind = (types or {}).get(row["unique_id"])
    if not kind:
        return ""
    return (f"<p class='sa-small'><b>Run by:</b> listed in the dataset as {esc(kind)}. "
            "Not checked.</p>")


def size_html(row) -> str:
    beds, doctors = facility_size(row["unique_id"])
    parts = ([f"{beds:,} beds"] if beds else []) + ([f"{doctors:,} doctors"] if doctors else [])
    if not parts:
        return ("<p class='sa-small'><b>Size:</b> the record gives no number of beds or "
                "doctors.</p>")
    return (f"<p class='sa-small'><b>Size, as the record gives it:</b> {', '.join(parts)}. "
            "Reported by the facility or its listing, not checked, and not used to order "
            "this list.</p>")


def well_rated(row, service) -> bool:
    """The two highest ratings: 3 or 4 of 4 mentioned for a mother; Level II or III clues
    for a newborn."""
    result = score(row, service)
    if result:
        return result[0] >= result[1] - 1
    return row["evidence_tier"] in ("1_strong", "2_medium")


@st.cache_data(ttl=3600, show_spinner=False)
def search_places(term: str) -> pd.DataFrame:
    # term is restricted to letters, digits, spaces, dots and hyphens before this call.
    query = f"""
        SELECT place_type, place_name, district, state, pin, lat, lon, n_facilities
        FROM {SCHEMA}.place_lookup
        WHERE search_key LIKE '%{term}%'
        ORDER BY CASE WHEN search_key = '{term}' THEN 0
                      WHEN search_key LIKE '{term}%' THEN 1 ELSE 2 END,
                 CASE place_type WHEN 'district' THEN 0 WHEN 'city' THEN 1 ELSE 2 END,
                 place_name
        LIMIT 25
    """
    return run_query(query)


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_pin(pin: str) -> pd.DataFrame:
    query = f"""
        SELECT pin, pin_lat, pin_lon, district_example, state_example, n_districts
        FROM {SCHEMA}.pin_lookup
        WHERE pin = '{pin}' AND pin_lat IS NOT NULL
    """
    return run_query(query)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def has(value) -> bool:
    """True when a value from the data is present (not None, NaN or blank)."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def split_list(value) -> list:
    if not has(value):
        return []
    return [part.strip() for part in str(value).split(SEP) if part.strip()]


def short(text, limit: int = 220) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def km(value) -> str:
    return f"{float(value):.0f} km" if has(value) else "an unknown distance"


def counts_24h(level) -> bool:
    """24-hour wording that counts for the rating: emergency care, or the service itself."""
    return level in ("service_24h", "emergency_24h")


def says_24h(level) -> bool:
    return level in ("service_24h", "emergency_24h", "open_24h")


def score(row, service):
    """(mentioned, out of) for services rated on a list of conditions; None for NICU."""
    items = COUNTED.get(service)
    if not items:
        return None
    n = 0
    for _, column in items:
        if column == "__hours__":
            n += 1 if counts_24h(row["hours_level"]) else 0
        else:
            n += 1 if has(row[column]) else 0
    return n, len(items)


def rating_label(row, service) -> str:
    result = score(row, service)
    if result:
        return f"{result[0]} of {result[1]} mentioned"
    return NICU_LEVELS.get(row["evidence_tier"], ("NICU named, no details", ""))[0]


def low(label: str) -> str:
    """A row label as it reads in the middle of a sentence."""
    first = label.split(" ")[0]
    if label.startswith("C-section") or (len(first) > 1 and first.isupper()):   # "NICU ..."
        return label
    return label[:1].lower() + label[1:]


def top_label(service) -> str:
    """How the best rating reads for this service."""
    items = COUNTED.get(service)
    if items:
        return f"all {len(items)} mentioned"
    return "NICU with ventilator or specialist"


def top_phrase(service) -> str:
    items = COUNTED.get(service)
    if items:
        names = [low(label) for label, _ in items]
        return "mentions " + ", ".join(names[:-1]) + " and " + names[-1]
    return "mentions a NICU with a ventilator or a newborn specialist"


def answers(service: str) -> dict:
    """What the user ticked on the questions page, turned into things to look for."""
    rule = situation_rule(service)
    needs, wants, notes, ticked = list(rule["needs"]), list(rule["wants"]), [], []
    for _, options in offered(service):
        for key, label, need, want, note in options:
            if st.session_state.get(f"ask_{service}_{key}"):
                ticked.append(label)
                needs += [n for n in need if n not in needs]
                wants += [w for w in want if w not in wants]
                if note:
                    notes.append(note)
    wants = [w for w in wants if w not in needs]
    return {"needs": needs, "wants": wants, "notes": notes, "ticked": ticked}


def record_has(row, item: str):
    """(mentioned?, quoted sentence or None) for one thing a record can be asked for."""
    column = ASK_ITEMS[item][1]
    if column == "__hours__":
        ok = counts_24h(row["hours_level"])
        return ok, (row["hours_quote"] if ok else None)
    if column == "__nicu__":
        ok = row["nicu_level"] in ("level3", "level2", "unknown")
        quote = row["sig_newborn"] if has(row["sig_newborn"]) else row["nicu_quote"]
        return ok, (quote if ok else None)
    value = row[column] if column in row else None
    return has(value), (value if has(value) else None)


def meets_needs(row, needs: list) -> bool:
    return all(record_has(row, item)[0] for item in needs)


def item_names(items: list) -> str:
    """'blood, 24-hour emergency care' for use inside a sentence."""
    return ", ".join(low(ASK_ITEMS[item][0]) for item in items)


def asked_html(row: pd.Series, service: str) -> str:
    """On the card: of the things the user's answers call for, which the record mentions."""
    told = answers(service)
    items = told["needs"] + told["wants"]
    if not items:
        return ""
    yes = [item for item in items if record_has(row, item)[0]]
    no = [item for item in items if item not in yes]
    parts = ["<b>For what you told us.</b>"]
    if yes:
        parts.append(f" Mentioned: {esc(item_names(yes))}.")
    if no:
        parts.append(f" <b>Not mentioned: {esc(item_names(no))}.</b> Ask when you call.")
    return f'<p class="sa-fac-sum" style="margin-top:6px">{"".join(parts)}</p>'


# What to ask on the phone when the record is silent about something. These are plain
# questions built from what the record does not mention; they are not clinical advice.
PHONE_QUESTIONS = {
    "C-section": "Can you do a C-section now, if it is needed?",
    "Blood": "Do you have blood available now?",
    "24-hour emergency care": "Is a doctor on duty now for emergencies?",
    "Newborn care": "Do you have a newborn care unit?",
    "A NICU or special newborn care unit": "Do you have a NICU or a special newborn care unit?",
    "Newborn ventilation": "Do you have a ventilator for a newborn?",
    "Newborn phototherapy": "Do you have phototherapy for a newborn?",
    "Warmer or incubator": "Do you have a warmer or an incubator?",
    "Paediatric intensive care": "Do you have paediatric intensive care?",
    "Ambulance": "Can you send an ambulance?",
    "Maternal-fetal medicine": "Is there a specialist for high-risk pregnancy?",
    "Normal delivery": "Do you support normal delivery?",
    "Pain relief in labour (epidural)": "Do you offer an epidural for pain relief in labour?",
    "Checkups during pregnancy": "Do you do checkups during pregnancy?",
    "Care after the birth": "Do you give care after the birth?",
    "Vaccination": "Do you vaccinate babies?",
    "Ultrasound": "Do you do ultrasound scans?",
    "Laboratory tests": "Do you do blood tests?",
}


def phone_questions(row, service: str, limit: int = 4) -> list:
    """Questions for the phone call: can they take the patient, then what the record
    does not mention, starting with what the user's answers call for."""
    urgent = st.session_state.get("role", "urgent") == "urgent"
    if urgent:
        first = ("Can you take the baby now, and is a place free in the newborn unit?"
                 if service == "nicu" else "Can you admit her now?")
    else:       # planning ahead: nobody is travelling yet
        first = ("Do you take newborns brought from outside, at any hour?"
                 if service == "nicu" else "Do you admit women for delivery at any hour?")
    missing = []
    told = answers(service)
    for item in told["needs"] + told["wants"]:
        if not record_has(row, item)[0]:
            missing.append(ASK_ITEMS[item][0])
    if service == "nicu":
        if row["evidence_tier"] == "4_basic":
            missing.append("A NICU or special newborn care unit")
        elif row["evidence_tier"] not in ("1_strong", "2_medium"):
            missing.append("Newborn ventilation")
    else:
        missing += [r[0] for r in checklist(row, service) if r[4] and r[3] != "yes"]
    questions = [first]
    if service != "nicu" and csection_limit(row):
        questions.append("Can you do an emergency C-section" + (" now?" if urgent else "?"))
    for label in missing:
        text = PHONE_QUESTIONS.get(label)
        if text and not urgent:
            text = text.replace(" now", "")
        if text and text not in questions:
            questions.append(text)
    if row["list_section"] != "1_main":
        questions.insert(1, "What is your exact address?")
    return questions[:limit]


def questions_html(row, service: str, cls: str = "sa-askq") -> str:
    return (f'<ol class="{cls}">' + "".join(f"<li>{esc(q)}</li>" for q in phone_questions(row, service))
            + "</ol>")


MANY_PHONES = 20     # about the top 3% of records on the two lists; see mixed_record()


def mixed_record(row) -> int:
    """The number of phone numbers when a record lists so many that it probably holds
    other facilities' details; 0 otherwise. The number of web pages is not used: the
    dataset keeps at most 50 per record and a third of the records reach 30."""
    n_phones = int(row["n_phone_numbers"]) if has(row["n_phone_numbers"]) else 0
    return n_phones if n_phones >= MANY_PHONES else 0


def mixed_words(row) -> str:
    """What the record says about its own size next to its phone numbers. Many numbers are
    normal for a very large hospital and odd for a small one, so both facts are shown."""
    phones = mixed_record(row)
    beds, _ = facility_size(row["unique_id"])
    size = f"gives {beds:,} beds" if beds else "gives no number of beds"
    return (f"lists {phones} phone numbers and {size}. So many numbers are normal for a "
            "very large hospital. For a small one they suggest that details of other "
            "facilities were mixed in")


def csection_limit(row) -> str:
    """The limiting word when the record's C-section wording is narrower than an emergency
    C-section ('uncomplicated LSCS'); an empty string otherwise."""
    text = str(row["sig_csection"]).lower() if has(row["sig_csection"]) else ""
    found = re.search(r"uncomplicated|elective|planned|non[- ]?emergency", text)
    return found.group(0) if found else ""


def call_box(row, service: str) -> str:
    """For the first choice and the backup: one large button to call, and what to ask."""
    if not has(row["phone"]):
        return ('<div class="sa-callbox"><p style="margin-top:0">No usable phone number in the '
                "record. If you find one, ask:</p>" + questions_html(row, service, "") + "</div>")
    return ('<div class="sa-callbox">'
            f'<a class="sa-call" href="tel:{esc(row["phone"])}">Call now</a>'
            "<p>Ask on the phone</p>" + questions_html(row, service, "") + "</div>")


def map_url(row) -> str:
    """A map link, only when the location is confirmed."""
    if row["list_section"] == "1_main" and has(row["lat"]) and has(row["lon"]):
        return ("https://www.google.com/maps/search/?api=1&query="
                f"{float(row['lat']):.5f},{float(row['lon']):.5f}")
    return ""


def share_text(row) -> str:
    """Plain text to send to a relative or a driver. Only what the record gives."""
    address = ", ".join(str(part) for part in [row["address_line1"], row["city"], row["state"], row["pin"]]
                        if has(part))
    lines = [str(row["name"])]
    if address:
        lines.append(address)
    if has(row["phone"]):
        lines.append(f"Phone: {row['phone']}")
    if map_url(row):
        lines.append(f"Map: {map_url(row)}")
    lines.append("Call before travelling. These details come from a public record and may be "
                 "out of date.")
    return "\n".join(lines)


def share_html(row) -> str:
    text = url_quote(share_text(row))
    return ('<p class="sa-share"><b>Send these details to someone</b><br>'
            f'<a href="https://wa.me/?text={text}" target="_blank" rel="noopener">WhatsApp</a>'
            f'<a href="sms:?&body={text}">Text message</a><br>'
            "Sends the name, address, phone and map link. Works on a phone; nothing is sent "
            "until you choose who to send it to.</p>")


CHILDREN_ONLY = r"child|p(a)?ediatr|neonat|newborn|\bkids?\b"
MOTHER_FITS = (r"matern|wom[ae]n|mother|gyn|obstet|fertil|\bivf\b|multi|general|"
               r"super.?special|medical college|medical sciences|nursing home")


# Where a rule comes from. A small "i" next to a statement opens its source in a new tab.
SOURCES = {
    "who": ("WHO, Monitoring emergency obstetric care: a handbook (2009)",
            "https://www.who.int/publications-detail-redirect/9789241547734"),
    "fru": ("First Referral Units in India: caesarean section, newborn care and blood "
            "storage, available 24 hours",
            "https://pmc.ncbi.nlm.nih.gov/articles/5144115"),
    "pmsma": ("Ministry of Health and Family Welfare, India: high-risk pregnancies are linked "
              "to the nearest First Referral Unit",
              "https://mohfw.gov.in/press-info/7478"),
    "fbnc": ("National Health Mission, Facility Based Newborn Care Operational Guidelines 2025",
             "https://nhm.gov.in/images/pdf/programmes/CH-Programmes/FBNC/"
             "FBNC%20Operational%20Guidelines%202025.pdf"),
    "nnf": ("National Neonatology Forum of India, accreditation form for level IIIB units",
            "https://www.nnfi.org/assests/upload/download-pdf/FINAL_FORM_LEVEL_IIIB_2022-23.pdf"),
}


def src(key: str) -> str:
    """The small "i": a link to the source, with the source's name as its label."""
    label, url = SOURCES[key]
    return (f'<a class="sa-i" href="{url}" target="_blank" rel="noopener" '
            f'title="Source: {esc(label)} (opens in a new tab)" '
            f'aria-label="Source: {esc(label)} (opens in a new tab)">i</a>')


def basis_html(service: str) -> str:
    """One line saying which standards the rating follows, each with its source."""
    if service == "nicu":
        return ("The four groups follow the Indian guidelines for newborn care in "
                f"facilities{src('fbnc')} and the levels of the National Neonatology Forum "
                f"of India{src('nnf')}. They are clues from wording, not official gradings.")
    return ("The four items follow the WHO description of emergency care at birth"
            f"{src('who')} and the Indian standard for First Referral Units{src('fru')}.")


def name_points_elsewhere(name, service: str = "") -> str:
    """'an eye hospital' when the name points to another specialty, else an empty string.
    For a mother, a name that points only to children counts as another specialty."""
    text = str(name).lower() if has(name) else ""
    if (text and service and service != "nicu" and re.search(CHILDREN_ONLY, text)
            and not re.search(MOTHER_FITS, text)):
        return "a children's hospital"
    if not text or re.search(NAME_FITS, text):
        return ""
    for pattern, what in OTHER_SPECIALTY:
        if re.search(pattern, text):
            return what
    if "hospital" not in text:
        for pattern, what in OTHER_SPECIALTY_NO_HOSPITAL:
            if re.search(pattern, text):
                return what
    return ""


def chance(row, service) -> float:
    """Stand-in for the chance that a facility can help. It is not a measured probability.

    A mother: items mentioned / (items + 1), so a full record gives 0.8 and never 1.
    A newborn: a fixed value for each level of clues.
    """
    result = score(row, service)
    if result:
        return result[0] / (result[1] + 1)
    return NICU_CHANCE.get(row["evidence_tier"], NICU_CHANCE["3_weak"])


def distances_between(lat, lon) -> np.ndarray:
    """Straight-line km between every pair of facilities. Unknown positions give infinity."""
    lat = np.radians(np.asarray(lat, dtype=float))[:, None]
    lon = np.radians(np.asarray(lon, dtype=float))[:, None]
    a = (np.sin((lat - lat.T) / 2) ** 2
         + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2)
    between = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.where(np.isnan(between), np.inf, between)


def plan_costs(distance, p, between, penalty):
    """Expected travel, in km, of the plan "go to j first, then to the best backup".

    after_fail[j] = the smaller of
                      penalty                              (stop: care is not reached)
                      between[j, l] + (1 - p[l]) * penalty (go on to the best backup l)
    cost[j]       = distance[j] + (1 - p[j]) * after_fail[j]

    Returns cost, backup (-1 where stopping is no worse than any backup) and after_fail.
    """
    n = len(distance)
    if n < 2:
        after_fail = np.full(n, float(penalty))
        return distance + (1 - p) * after_fail, np.full(n, -1), after_fail
    onward = between + (1 - p)[None, :] * penalty
    np.fill_diagonal(onward, np.inf)
    backup = onward.argmin(axis=1)
    best_onward = onward[np.arange(n), backup]
    useful = best_onward < penalty
    after_fail = np.where(useful, best_onward, float(penalty))
    return distance + (1 - p) * after_fail, np.where(useful, backup, -1), after_fail


def decide(distance, p, between, penalty=PENALTY_KM, break_even=True) -> dict:
    """Pick the first choice and its backup. Facilities must be sorted nearest first.

    break_even is the lowest chance, in steps of 0.01, at which going to the nearest
    facility first becomes the cheapest plan and stays so for every higher chance. It is
    None when the nearest facility is already the first choice.
    """
    distance = np.asarray(distance, dtype=float)
    p = np.asarray(p, dtype=float)
    cost, backup, after_fail = plan_costs(distance, p, between, penalty)
    first = int(cost.argmin())
    result = {"first": first, "backup": int(backup[first]), "cost": cost, "p": p,
              "backups": backup, "after_fail": after_fail, "break_even": None,
              # False when even the best plan costs more than not reaching care is counted as
              "worth_it": bool(cost[first] <= penalty)}
    if first != 0 and break_even:
        break_even = 1.0
        for step in range(100, -1, -1):
            trial = p.copy()
            trial[0] = step / 100
            trial_cost = plan_costs(distance, trial, between, penalty)[0]
            if int(trial_cost.argmin()) != 0:
                break
            break_even = step / 100
        result["break_even"] = break_even
    return result


def other_facility_named(quote: str, facility_name: str) -> str:
    """If a quote names a facility whose name shares no word with this one, return that name.

    Records sometimes carry sentences that describe a different facility.
    """
    own = {w for w in re.findall(r"[a-z]+", str(facility_name).lower())
           if len(w) >= 4 and w not in GENERIC_NAME_WORDS}
    if not own:
        return ""
    pattern = (r"((?:[A-Z][A-Za-z&.'-]+ ){1,4})"
               r"(Hospitals?|Centre|Center|Clinic|Nursing Home|Institute)\b")
    for match in re.finditer(pattern, str(quote)):
        words = {w for w in re.findall(r"[a-z]+", match.group(1).lower())
                 if len(w) >= 4 and w not in GENERIC_NAME_WORDS}
        if words and not (words & own):
            return (match.group(1) + match.group(2)).strip()
    return ""


def go(mode, service_label=None, place=None):
    """Button callback: switch screen, optionally pre-filling a service and place."""
    st.session_state["mode"] = mode
    st.session_state["edit_place"] = False
    if mode in MODES:
        st.session_state["role"] = mode
    if service_label:
        st.session_state["service_label"] = service_label
        st.session_state["situation"] = None      # the home page examples carry no situation
    if place:
        st.session_state["place_query"] = place
        st.session_state["place_confirmed"] = clean_term(place)


def change_answers():
    """Button callback on the results: go to the questions and come straight back after."""
    go("needs")
    st.session_state["back_to_results"] = True


def needs_done(role):
    """Button callback for "Continue" on the questions page.

    Walking through the steps, the next step is always the place, even when a place is
    remembered from an earlier search: the box shows it and the user confirms or changes
    it. Only someone who came from the results to change an answer returns to them directly.
    """
    direct = st.session_state.pop("back_to_results", False)
    go(role)
    st.session_state["edit_place"] = not direct


def edit_place():
    """Button callback: from the results, go one step back to the question about the place."""
    st.session_state["edit_place"] = True


def place_done():
    """Button callback: the place was (re)entered, so the results can show again."""
    st.session_state["edit_place"] = False


def clear_place():
    """Button callback: empty the place box. With no place, step 3 is shown again."""
    st.session_state["place_query"] = ""
    st.session_state["place_confirmed"] = None
    st.session_state.pop("geo_query", None)
    st.session_state.pop("geo_error", None)


def clear_question():
    """Button callback: empty the question box on the Genie page."""
    st.session_state["genie_question"] = ""


def pick_patient(label):
    """Button callback for step 2: remember who the patient is, then ask what is happening."""
    st.session_state["service_label"] = label
    st.session_state["mode"] = "needs"
    st.session_state["edit_place"] = False
    st.session_state["back_to_results"] = False


def pick_role(role):
    """Button callback for step 1: remember what the user needs to do, then ask the service."""
    st.session_state["role"] = role
    st.session_state["mode"] = "service"
    st.session_state["situation"] = None
    st.session_state["back_to_results"] = False


def set_view():
    """Radio callback: switch between the three views of the results."""
    go(st.session_state["view_radio"])


def confirm_place(term):
    """Button callback: the user picked which of several matching places they mean."""
    st.session_state["place_confirmed"] = term
    st.session_state["edit_place"] = False


# ----------------------------------------------------------------------------
# Shared inputs
# ----------------------------------------------------------------------------

def place_label(row: pd.Series) -> str:
    name, state = str(row["place_name"]), str(row["state"]) if has(row["state"]) else ""
    if row["place_type"] == "district":
        return f"{name} district, {state}"
    if row["place_type"] == "city":
        n = int(row["n_facilities"]) if has(row["n_facilities"]) else 0
        return f"{name} (town or city), {state} - {n} facilities in the dataset"
    district = str(row["district"]) if has(row["district"]) else "unknown"
    pin = f", PIN {row['pin']}" if has(row["pin"]) else ""
    return f"{name} - locality in {district} district, {state}{pin}"


def clean_term(query) -> str:
    """A typed place name, reduced to what the place search accepts."""
    return re.sub(r"[^a-z0-9 .\-]", "", str(query or "").lower()).strip()


def resolve_place(query) -> dict:
    """Work out what the typed starting place means. Draws nothing.

    Returns {'status': 'empty'}, {'status': 'error', 'message': ...},
    {'status': 'ok', 'location': {...}, 'note': ...} or
    {'status': 'choices', 'term': ..., 'places': DataFrame, 'labels': [...]}.
    """
    query = str(query or "").strip()
    if not query:
        return {"status": "empty"}
    from_device = query == st.session_state.get("geo_query")

    coords = re.fullmatch(r"(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)", query)
    if coords:
        lat, lon = float(coords.group(1)), float(coords.group(2))
        if not (6 <= lat <= 38 and 68 <= lon <= 98):
            if from_device:
                return {"status": "error", "message":
                        "Your current location is outside India, so there is nothing to "
                        "search near it. Type a place in India instead."}
            return {"status": "error", "message":
                    "These coordinates are outside India. Enter latitude first, then "
                    "longitude, for example 25.61, 85.14."}
        if from_device:
            label, short_name = "your current location", "your location"
        else:
            label, short_name = f"the point {lat:.4f}, {lon:.4f}", f"{lat:.2f}, {lon:.2f}"
        return {"status": "ok", "note": f"Searching from {label}.",
                "location": {"label": label, "short": short_name, "lat": lat, "lon": lon}}

    if re.fullmatch(r"[0-9]{6}", query):
        found = lookup_pin(query)
        if found.empty:
            return {"status": "error", "message":
                    f"PIN code {query} is not in the India Post directory, or it has no "
                    "position there. Check the number or type the place name instead."}
        row = found.iloc[0]
        label = f"PIN code {query}"
        if has(row["district_example"]):
            label += f" ({str(row['district_example']).title()}, {str(row['state_example']).title()})"
        note = f"Searching from the centre of {label}. Distances are approximate."
        if has(row["n_districts"]) and int(row["n_districts"]) > 1:
            note += " This PIN code is shared by more than one district."
        return {"status": "ok", "note": note,
                "location": {"label": label, "short": f"PIN {query}",
                             "lat": float(row["pin_lat"]), "lon": float(row["pin_lon"])}}

    term = clean_term(query)
    if len(term) < 3:
        return {"status": "error", "message":
                "Type at least 3 letters of the place name, or a 6-digit PIN code."}
    places = search_places(term)
    if places.empty:
        return {"status": "error", "message":
                f"No place called '{query}' was found. Check the spelling, try the district "
                "name, or enter a PIN code."}
    labels = [place_label(places.iloc[i]) for i in range(len(places))]
    return {"status": "choices", "term": term, "places": places, "labels": labels}


def chosen_place(found: dict) -> dict:
    """The location for the place the user picked among several matches (first by default)."""
    index = st.session_state.get(f"place_choice_{found['term']}", 0)
    if not isinstance(index, int) or not 0 <= index < len(found["labels"]):
        index = 0
    row = found["places"].iloc[index]
    name = str(row["place_name"])
    short_name = f"{name} district" if row["place_type"] == "district" else name
    note = "Searching from the centre of this place. Distances are approximate."
    if row["place_type"] == "district":
        note += " A district is large: for closer distances, enter your town, village or PIN code."
    return {"note": note,
            "location": {"label": found["labels"][index], "short": short_name,
                         "lat": float(row["lat"]), "lon": float(row["lon"])}}


def place_is_settled(found: dict) -> bool:
    """True when we know where to search from, so the results can be shown."""
    if found["status"] == "ok":
        return True
    if found["status"] == "choices":
        return (len(found["labels"]) == 1
                or st.session_state.get("place_confirmed") == found["term"])
    return False


def pick_among(found: dict) -> None:
    """Dropdown to choose among several places with the same name."""
    labels = found["labels"]
    st.selectbox(
        f"{len(labels)} places match. Which one do you mean?",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key=f"place_choice_{found['term']}",
    )


def fetch(service, location, radius_km, sort, limit):
    try:
        with st.spinner("Searching facilities..."):
            return get_recommendations(
                service, location["lat"], location["lon"], float(radius_km), sort, int(limit)
            )
    except Exception as exc:  # show the problem instead of a blank page
        st.error(f"The search failed: {exc}")
        return None


def by_match(df: pd.DataFrame, service: str) -> pd.DataFrame:
    """Records that mention more of what the user ticked come first, then nearest.
    Unconfirmed locations stay last."""
    told = answers(service)
    items = told["needs"] + told["wants"]
    if df.empty or not items:
        return df
    matched = df.apply(lambda r: sum(record_has(r, item)[0] for item in items), axis=1)
    return (df.assign(_m=-matched).sort_values(["list_section", "_m", "distance_km"])
              .drop(columns="_m"))


def by_rating(df: pd.DataFrame, service: str) -> pd.DataFrame:
    """Most mentioned first, then nearest. Unconfirmed locations stay last."""
    if df.empty:
        return df
    counted = df.apply(lambda r: (score(r, service) or (0, 0))[0], axis=1)
    return (df.assign(_n=-counted)
              .sort_values(["list_section", "_n", "evidence_tier", "distance_km"])
              .drop(columns="_n"))


# ----------------------------------------------------------------------------
# Facility card
# ----------------------------------------------------------------------------

def flag_other(text, facility_name) -> str:
    other = other_facility_named(text, facility_name)
    if other:
        return (f' <span class="sa-flag">(names "{esc(other)}": this sentence may describe '
                "another facility)</span>")
    return ""


def checklist(row: pd.Series, service: str) -> list:
    """Rows for the 'what the record says' table: (label, status, detail, kind, counted)."""
    rows = []
    silent = "The record is silent. Ask when you call."
    name = row["name"]

    def quote(text) -> str:
        return f'"{esc(short(text))}"{flag_other(text, name)}'

    for label, column in COUNTED.get(service, []):
        if column == "__hours__":
            level = row["hours_level"]
            if counts_24h(level):
                rows.append((label, "Mentioned", quote(row["hours_quote"]), "yes", True))
            elif level == "open_24h":
                rows.append((label, "Not mentioned",
                             f'The record only says the facility is open: {quote(row["hours_quote"])}. '
                             "Being open is not the same as emergency care, so this is not counted.",
                             "no", True))
            else:
                rows.append((label, "Not mentioned", silent, "no", True))
        elif has(row[column]):
            rows.append((label, "Mentioned", quote(row[column]), "yes", True))
        else:
            rows.append((label, "Not mentioned", silent, "no", True))

    if service == "nicu":
        level_name = NICU_LEVELS.get(row["evidence_tier"], ("NICU named, no details", ""))[0]
        detail = quote(row["nicu_quote"]) if has(row["nicu_quote"]) else silent
        rows.append(("Newborn unit", level_name, detail, "yes", True))

    for label, column in INFO_ROWS.get(service, []):
        if column == "__hours_any__":
            value = row["hours_quote"] if says_24h(row["hours_level"]) else None
        else:
            value = row[column]
        if has(value):
            rows.append((label, "Mentioned", quote(value), "yes", False))
        else:
            rows.append((label, "Not mentioned", silent, "no", False))
    return rows


def checklist_html(row: pd.Series, service: str) -> str:
    """The full table with the quoted wording. Shown in the popup."""
    body = []
    for name, status, detail, kind, counted in checklist(row, service):
        tag = "" if counted else '<span class="nc">for information, not counted</span>'
        cls = " no" if kind == "no" else ""
        body.append(
            f'<tr><th scope="row">{esc(name)}{tag}</th>'
            f'<td class="s{cls}">{esc(status)}</td><td class="d{cls}">{detail}</td></tr>'
        )
    table = (f'<table class="sa-table"><caption>What the record says: '
             f'{esc(rating_label(row, service))}</caption><tbody>{"".join(body)}</tbody></table>')
    told = answers(service)
    asked = told["needs"] + told["wants"]
    if not asked:
        return table
    extra = []
    for item in asked:
        ok, quote = record_has(row, item)
        detail = (f'"{esc(short(quote))}"{flag_other(quote, row["name"])}' if ok and has(quote)
                  else "The record is silent. Ask when you call.")
        cls = "" if ok else " no"
        extra.append(f'<tr><th scope="row">{esc(ASK_ITEMS[item][0])}</th>'
                     f'<td class="s{cls}">{"Mentioned" if ok else "Not mentioned"}</td>'
                     f'<td class="d{cls}">{detail}</td></tr>')
    return (table + '<table class="sa-table" style="margin-top:18px"><caption>For what you '
            f'told us</caption><tbody>{"".join(extra)}</tbody></table>')


def summary_html(row: pd.Series, service: str) -> str:
    """Always visible on the card: the rating, what is mentioned and what is missing.

    What is missing stays outside the popup on purpose: it is what the user has to ask
    about on the phone.
    """
    label = esc(rating_label(row, service))
    if service == "nicu":
        parts = [f"<b>{label}.</b>"]
        if has(row["nicu_quote"]):
            parts.append(f' The record says: "{esc(short(row["nicu_quote"], 120))}".')
        tier = row["evidence_tier"]
        if tier == "4_basic":
            parts.append(" <b>No NICU is named.</b> Ask when you call.")
        elif tier not in ("1_strong", "2_medium"):
            parts.append(" <b>No equipment or staff detail is mentioned.</b> Ask when you call.")
        return f'<p class="sa-fac-sum">{"".join(parts)}</p>'

    rows = [r for r in checklist(row, service) if r[4]]
    def names(kind: str) -> str:      # "C-section" must not break at its hyphen
        listed = ", ".join(esc(low(r[0])) for r in rows if r[3] == kind)
        return listed.replace("C-section", '<span style="white-space:nowrap">C-section</span>')

    parts = [f"<b>{label}.</b>"]
    if names("yes"):
        parts.append(f" Mentioned: {names('yes')}.")
    if names("no"):
        parts.append(f" <b>Not mentioned: {names('no')}.</b> Ask when you call.")
    if csection_limit(row):
        parts.append(f' <b>The record says "{esc(csection_limit(row))}" C-section.</b> Ask '
                     "whether they also do emergency C-sections.")
    return f'<p class="sa-fac-sum">{"".join(parts)}</p>'


def card_html(row: pd.Series, number: int, service: str, prominent: bool = False) -> str:
    uncertain = row["list_section"] != "1_main"
    if uncertain:
        dist = "Location not confirmed"
    else:
        dist = f"About {km(row['distance_km'])}, straight line"
    address = ", ".join(
        esc(part) for part in [row["address_line1"], row["city"], row["state"], row["pin"]]
        if has(part)
    )
    if has(row["phone"]):
        phone = f'Phone: <a href="tel:{esc(row["phone"])}">{esc(row["phone"])}</a>'
    else:
        phone = ('Phone: <span style="font-weight:400;font-style:italic">none usable in the '
                 "record</span>")
    parts = [
        '<div class="sa-fac-head">',
        f'<h3 class="sa-fac-name">{number}. {esc(row["name"])}</h3>',
        f'<p class="sa-fac-dist">{dist}</p></div>',
    ]
    if address:
        parts.append(f'<p class="sa-fac-addr">{address}</p>')
    parts.append(f'<p class="sa-fac-phone">{phone}</p>')
    if uncertain:
        parts.append(
            '<div class="sa-note" style="margin-bottom:14px">The record gives two locations. '
            f"Its postal code places it about {km(row['distance_km'])} away; its map position "
            f"places it about {km(row['distance_km_by_coords'])} away. Ask for the address "
            "when you call.</div>"
        )
    elsewhere = name_points_elsewhere(row["name"], service)
    if elsewhere:
        parts.append(
            f'<div class="sa-note" style="margin-bottom:12px"><b>Check this one.</b> The name '
            f"suggests {elsewhere}, yet the record mentions "
            f"{'newborn' if service == 'nicu' else 'birth'} care. The record may contain "
            "another facility's details. Confirm by phone that they provide it.</div>")
    phones = mixed_record(row)
    if phones:
        parts.append(
            '<div class="sa-note" style="margin-bottom:12px"><b>Check this record.</b> It '
            f"{mixed_words(row)}. Confirm by phone what this facility offers.</div>")
    if not uncertain and row["location_quality"] in ("conflict_coords_confirmed",
                                                      "conflict_pin_confirmed"):
        used_ = ("map position" if row["location_quality"] == "conflict_coords_confirmed"
                 else "postal code")
        parts.append(
            '<div class="sa-note" style="margin-bottom:12px"><b>Check the address.</b> The '
            "postal code and the map position in this record point to different places. "
            f"The distance shown uses the {used_}. Ask for the address when you call.</div>")
    if prominent:
        parts.append(call_box(row, service))
    parts.append(summary_html(row, service))
    parts.append(asked_html(row, service))
    return "".join(parts)


def show_details(row: pd.Series, service: str) -> None:
    """Everything behind one card: the quoted wording, the location check and the sources."""
    if row["list_section"] == "1_main":
        dist = f"About {km(row['distance_km'])}, straight line"
    else:
        dist = "Location not confirmed"
    if has(row["phone"]):
        phone = f'Phone: <a href="tel:{esc(row["phone"])}">{esc(row["phone"])}</a>'
    else:
        phone = "Phone: none usable in the record"
    show_html(f'<p class="sa-small" style="margin-bottom:14px">{dist} &nbsp;&middot;&nbsp; {phone}</p>'
              + checklist_html(row, service)
              + '<p class="sa-small" style="margin-top:10px">"Mentioned" means the record says '
              "it. Nobody has confirmed it. \"Not mentioned\" means the record is silent. "
              + basis_html(service) + "</p>"
              "<p class='sa-small' style='margin-top:14px'><b>What to ask when you call</b></p>"
              + questions_html(row, service) + share_html(row)
              + '<hr class="sa-side-rule" style="margin:18px 0 14px 0">')
    quotes = split_list(row["quotes"])
    lines = ["<p class='sa-small'><b>Everything the record says about this service</b></p><ul>"]
    for quote in quotes:
        field, _, text = quote.partition(": ")
        body = text or quote
        prefix = f"<i>{esc(FIELD_LABELS.get(field, field))}:</i> " if text else ""
        lines.append(f"<li>{prefix}{esc(body)}{flag_other(body, row['name'])}</li>")
    lines.append("</ul>")
    show_html("".join(lines))

    if row["list_section"] == "1_main":
        note = LOCATION_NOTES.get(row["location_quality"])
        if note:
            show_html(f"<p class='sa-small'><b>Location check:</b> {esc(note)}</p>")
    if st.session_state.get("role") != "urgent":      # for planning; a distraction in a hurry
        show_html(size_html(row) + operator_html(row))

    n_sources = int(row["n_sources"]) if has(row["n_sources"]) else 0
    n_phones = int(row["n_phone_numbers"]) if has(row["n_phone_numbers"]) else 0
    if n_sources >= 30 or n_phones >= 10:
        show_html(
            "<p class='sa-small'><b>Check the name when you call.</b> This record draws "
            f"on {n_sources} web pages and lists {n_phones} phone numbers, so it may "
            "mix in details from other facilities.</p>"
        )
    # some records hold the text "null" in this field
    if has(row["website"]) and str(row["website"]).strip().lower() not in ("null", "none", "nan"):
        site = str(row["website"])
        url = site if site.startswith("http") else f"https://{site}"
        show_html(f"<p class='sa-small'><b>Website listed in the record:</b> "
                  f"<a href='{esc(url)}'>{esc(site)}</a></p>")
    sources = split_list(row["sources"])
    items = "".join(f"<li><a href='{esc(u)}'>{esc(u)}</a></li>" for u in sources)
    show_html(
        f"<p class='sa-small'><b>Web pages this record was built from</b> ({n_sources} "
        f"in total, first {len(sources)} shown). The dataset does not say which "
        f"sentence came from which page.</p><ul>{items}</ul>"
    )


def language_for(state) -> str:
    """The language the driver card opens in, from the facility's state. English if unknown."""
    key = " ".join(str(state).split()).casefold() if has(state) else ""
    return STATE_LANGUAGE.get(STATE_FIX.get(key, key), "English")


def show_driver_card(row: pd.Series) -> None:
    """Something to hold up to a driver: where to go, in big letters."""
    uid = row["unique_id"]
    languages = list(DRIVER_SENTENCES.keys())
    start = language_for(row["state"])
    language = st.selectbox(
        "Language", languages, index=languages.index(start), key=f"driver_language_{uid}",
        format_func=lambda name: name if name == "English"
        else f"{DRIVER_SENTENCES[name][0]} ({name})",
    )
    _, take_me, emergency = DRIVER_SENTENCES[language]
    address = ", ".join(
        esc(part) for part in [row["address_line1"], row["city"], row["state"], row["pin"]]
        if has(part)
    )
    parts = ['<div class="sa-driver">',
             f'<p class="say" lang="{LANGUAGE_CODE[language]}">{esc(take_me)} {esc(emergency)}</p>']
    if language != "English":
        english = DRIVER_SENTENCES["English"]
        parts.append(f'<p class="en">{esc(english[1])} {esc(english[2])}</p>')
    parts.append(f'<p class="name">{esc(row["name"])}</p>')
    if address:
        parts.append(f'<p class="addr">{address}</p>')
    if has(row["phone"]):
        parts.append(f'<p class="tel">Hospital phone: <a href="tel:{esc(row["phone"])}">'
                     f'{esc(row["phone"])}</a></p>')
    parts.append("</div>")

    url = map_url(row)
    if url:
        rough = row["location_quality"] in ("pin_approx", "conflict_pin_confirmed")
        note = (" The pin is the centre of the postal code area, not the building. Use the "
                "address for the last stretch." if rough else "")
        parts.append(f'<p class="sa-driver-map"><a href="{url}" target="_blank" rel="noopener">'
                     f"Open in Google Maps</a>{note}</p>")
    else:
        parts.append('<div class="sa-note" style="margin-top:14px">The record gives two '
                     "locations for this facility, so there is no map link. Confirm the "
                     "address by phone before you set off.</div>")
    parts.append(share_html(row))

    numbers = "".join(f'<li><a href="tel:{number}">{number}</a> &nbsp;{esc(what)}</li>'
                      for number, what in EMERGENCY_NUMBERS)
    parts.append('<hr class="sa-side-rule" style="margin:18px 0 12px 0">'
                 '<p class="sa-small"><b>If you need an ambulance</b></p>'
                 f'<ul class="sa-driver-numbers">{numbers}</ul>'
                 '<p class="sa-fine">102 and 108 are not the same in every state (Rajasthan '
                 "uses 104). The name and address are shown exactly as the record gives them "
                 "and may be out of date. The sentences in other languages were written for "
                 "this tool and have not been checked by native speakers.</p>")
    show_html("".join(parts))


def show_card(row: pd.Series, number: int, service: str, prominent: bool = False) -> None:
    """A box with the essentials. Clicking anywhere on it opens the details in a popup."""
    with st.container():
        marker("fac" if hasattr(st, "dialog") else "facplain")
        show_html(card_html(row, number, service, prominent))
        label = "See what the record says and its sources"
        if hasattr(st, "dialog"):
            # Order matters for the styling: the details button first, the driver button
            # last (it is pinned to the bottom right corner of the box).
            if st.button(label, key=f"details_{number}_{row['unique_id']}"):
                st.dialog(str(row["name"]), width="large")(show_details)(row, service)
            if st.button("Show to driver", key=f"driver_{number}_{row['unique_id']}"):
                st.dialog("Show this to the driver", width="large")(show_driver_card)(row)
        else:  # older Streamlit without popups
            with st.expander(label):
                show_details(row, service)
            with st.expander("Show to driver"):
                show_driver_card(row)


def show_table(table: pd.DataFrame) -> None:
    """Full-width table that works on both older and newer Streamlit versions."""
    try:
        st.dataframe(table, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(table, hide_index=True, use_container_width=True)


def results_table(df: pd.DataFrame, service: str) -> pd.DataFrame:
    return pd.DataFrame({
        "#": range(1, len(df) + 1),
        "Facility": df["name"].values,
        "What the record says": [rating_label(r, service) for _, r in df.iterrows()],
        "24-hour wording": df["hours_level"].map({
            "service_24h": "This service, 24 hours",
            "emergency_24h": "24-hour emergency",
            "open_24h": "Open 24 hours (not counted)",
        }).fillna("Not mentioned").values,
        "Phone in record": df["phone"].values,
        "Distance (km)": df["distance_km"].round(0).values,
        "Town or city": df["city"].values,
        "Location": df["list_section"].map(
            lambda s: "Confirmed or approximate" if s == "1_main" else "Not confirmed").values,
    })


def summary_sentence(prefix: str, row, service) -> str:
    phone = esc(row["phone"]) if has(row["phone"]) else "none in the record"
    return (f"{prefix} is <b>{esc(row['name'])}</b>, about {km(row['distance_km'])} away. "
            f"Phone: {phone}.")


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------

def top_bar(back_to="home", back=None) -> None:
    """Brand and navigation. back = (callback, args) adds a "Back" button for one step back."""
    cols = st.columns([1, 2], vertical_alignment="center")
    show_html(f'<p class="sa-brand">{BRAND}</p>', cols[0])
    with cols[1].container():
        marker("navrow")
        if back_to == "home":
            if back is not None:
                st.button("Back", on_click=back[0], args=back[1], key="nav_back")
            st.button("Back to home", on_click=go, args=(None,), key="nav_home")
            st.button("How this tool decides", on_click=go, args=("about",), key="nav_about")
        elif back_to == "back":
            st.button("Back", on_click=go, args=(st.session_state.get("last_screen"),),
                      key="nav_back")


def call_first(service: str) -> None:
    ask = {
        "high_risk_delivery": "Ask whether they can take her today and whether emergency "
                              "surgery is available.",
        "nicu": "Ask whether a newborn bed is free and what level of care they give.",
    }[service]
    show_html(
        '<div class="sa-call"><p class="t">Call before travelling.</p>'
        f'<p class="d">Records can be out of date. {ask}</p></div>'
    )


def how_to_read(service: str) -> None:
    lines = [
        "<b>Mentioned</b> means the facility's record says it. Nobody has confirmed it.",
        "<b>Not mentioned</b> means the record is silent. The facility may still offer it.",
        basis_html(service),
    ]
    if service == "nicu":
        lines.append("<b>The four groups</b> (NICU with ventilator or specialist; special "
                     "newborn unit or CPAP; NICU named, no details; basic newborn care only) "
                     "come from the equipment or staff the record mentions. They are not an "
                     "official grading.")
    else:
        lines.append("<b>24-hour emergency care</b> is counted only when the record says so. "
                     "\"Open 24 hours\" alone is not counted.")
    lines.append("Distances are straight lines, not road distance.")
    body = "".join(f'<p class="sa-small" style="margin-bottom:12px">{t}</p>' for t in lines)
    show_html(f'<hr class="sa-side-rule"><p class="sa-side-h" style="margin-bottom:8px">'
              f'How to read this</p>{body}')
    with st.container():
        marker("links")
        st.button("How this tool decides", on_click=go, args=("about",), key="side_about")


def search_panel(mode: str, found: dict):
    """The cream panel on the right: what, where, which view and which order."""
    marker("side")
    show_html('<p class="sa-side-h">Your search</p>')
    label = st.selectbox("Who needs care", list(SERVICES.keys()), key="service_label")
    service = SERVICES[label]

    st.text_input("Starting from (place name, 6-digit PIN code, or latitude, longitude)",
                  key="place_query", placeholder="For example: Araria")
    with st.container():
        marker("links")
        st.button("Clear and enter another place", on_click=clear_place, key="side_clear")
    if found["status"] == "choices":
        if len(found["labels"]) > 1:
            pick_among(found)
        place = chosen_place(found)
    else:
        place = found
    st.caption(place["note"])
    location = place["location"]

    st.session_state["view_radio"] = mode
    st.radio("Show", list(MODES.keys()), format_func=lambda k: VIEW_LABELS[k],
             key="view_radio", on_change=set_view)

    options = {"mode": mode, "label": label, "service": service, "location": location}
    rated_first = "Most detail first" if service == "nicu" else "Most mentioned first"
    if mode == "urgent":
        pass    # the order is decided by the suggested plan, see view_urgent()
    elif mode == "compare":
        options["radius"] = st.number_input("Search within (km)", min_value=10, max_value=500,
                                            value=100, step=10, key="compare_radius")
        told_ = answers(service)
        choices = [rated_first, "Nearest first"]
        if told_["needs"] or told_["wants"]:
            choices.insert(0, "Best match to what you told us")
        if st.session_state.get("order_compare") not in choices:
            st.session_state.pop("order_compare", None)
        order = st.radio("List order", choices, key="order_compare")
        options["by_rating"] = order == rated_first
        options["by_match"] = order == "Best match to what you told us"
        options["only_24h"] = st.checkbox(
            "Only facilities whose record mentions 24-hour emergency care or 24-hour service",
            key="compare_only_24h")
        options["only_strong"] = st.checkbox(
            ("Only records that give detail about the newborn unit" if service == "nicu"
             else "Only records that mention at least 3 of the 4 items"),
            key="compare_only_strong",
            help="Leaves out records with weaker evidence. A facility left out may still "
                 "offer the service: its record just says less.")
        options["only_public"] = st.checkbox(
            "Only facilities listed as public or government", key="compare_only_public",
            help="Uses the dataset's own label for who runs the facility. Fewer than 5 in "
                 "100 facilities in the dataset carry it, so few will be left.")
    else:
        options["radius"] = st.number_input("Area radius (km)", min_value=10, max_value=300,
                                            value=100, step=10, key="area_radius")
    told = answers(service)
    situation = st.session_state.get("situation")
    if situation:
        show_html('<hr class="sa-side-rule"><p class="sa-side-h" style="margin-bottom:6px">'
                  f'Situation</p><p class="sa-small">{esc(situation)}</p>')
    listed = "".join(f"<li>{esc(text)}</li>" for text in told["ticked"])
    show_html('<hr class="sa-side-rule"><p class="sa-side-h" style="margin-bottom:6px">What '
              "you told us</p>"
              + (f'<ul class="sa-told">{listed}</ul>' if listed
                 else '<p class="sa-small">Nothing ticked.</p>'))
    with st.container():
        marker("links")
        st.button("Change these answers", on_click=change_answers, key="side_needs")
    how_to_read(service)
    return options


# ----------------------------------------------------------------------------
# Screens
# ----------------------------------------------------------------------------

def screen_home() -> None:
    """One screen, no scrolling on a desktop: five sections with the same gap between them."""
    st.session_state["last_screen"] = None
    marker("home")
    cols = st.columns([1, 2], vertical_alignment="center")
    show_html(f'<p class="sa-brand">{BRAND}</p>', cols[0])
    with cols[1].container():
        marker("navrow")
        st.button("How this tool decides, and what it cannot tell you",
                  on_click=go, args=("about",), key="home_about")

    with st.container():
        marker("sec")
        hero = st.columns([2, 1], gap="large", vertical_alignment="center")
        show_html(
            '<h1 class="sa-h1">Find where to take a mother or newborn, and what to ask before you go.</h1>'
            '<p class="sa-lede">Find maternal and newborn care in India. Every option shows '
            "what the facility's record says, and what it leaves out.</p>",
            hero[0],
        )
        with hero[1].container():
            marker("hero")
            picture = base64.b64decode(HERO_JPEG_BASE64)
            try:
                st.image(picture, use_container_width=True)
            except TypeError:  # older Streamlit
                st.image(picture, use_column_width=True)

    with st.container():
        marker("sec")
        marker("band")
        band = st.columns([3, 1], vertical_alignment="center")
        show_html(
            f'<h2 class="sa-h2 big sa-on-red">{MODES["urgent"]}</h2>'
            '<p class="sa-p sa-on-red" style="margin-top:4px">A first choice and a backup, '
            "each with a phone number to call.</p>",
            band[0],
        )
        with band[1].container():
            marker("toright")
            st.button("Start", on_click=pick_role, args=("urgent",), key="role_urgent",
                      type="primary")

    # One entry for everything that is not urgent. The list and the map are two views of the
    # same search, chosen on the results page.
    with st.container():
        marker("sec")
        marker("planbox")
        box = st.columns([3, 1], vertical_alignment="center")
        show_html(
            '<h2 class="sa-h2 big">Not urgent? Plan ahead</h2>'
            '<p class="sa-small" style="margin-top:4px">Compare options or check coverage: the '
            "full list for an area with what each record says, and a map of where the gaps "
            "are.</p>",
            box[0],
        )
        with box[1].container():
            marker("toright")
            st.button("Plan now", on_click=pick_role, args=("compare",), key="role_compare",
                      type="primary")

    examples = [
        ("A mother near Araria", "urgent", "A mother", "Araria"),
        ("A mother near Pune", "compare", "A mother", "Pune"),
        ("A newborn near Patna", "area", "A newborn", "Patna"),
    ]
    # Examples are a shortcut for someone exploring, so they are kept quiet: one line of
    # plain links under the two entries.
    with st.container():
        marker("sec")
        marker("quiet")
        show_html('<p class="sa-quiet">Or try an example:</p>')
        for i, (text, mode, service_label, place) in enumerate(examples):
            st.button(text, on_click=go, args=(mode, service_label, place), key=f"example_{i}")

    with st.container():
        marker("sec")
        show_html(
            '<p class="sa-fine" style="max-width:760px">Built on the Virtue Foundation dataset of about 10,000 '
            "facilities in India. Each record was put together from web pages, so every "
            "statement is a claim, not a confirmed fact. Always call the facility before "
            "sending a patient.</p>"
        )


PATHS = {
    "urgent": {
        "ask": "What is the situation?",
        "get": "You will get a suggested first choice and a backup, each with a phone number.",
        "where": "Where is the patient now?",
        "go": "Show the plan",
    },
    "compare": {
        "ask": "What are you planning for?",
        "get": "You will get the full list for an area and a map of where the gaps are.",
        "where": "Which area are you planning for?",
        "go": "Show the list",
    },
    "area": {
        "ask": "Whose care do you want to map?",
        "get": "You will get a map of facilities around a place, and where the gaps are.",
        "where": "Which area do you want to map?",
        "go": "Show the map",
    },
}


# Step 2 offers situations a person recognises, not our own categories. Each one fixes who
# the patient is; whether it is urgent was fixed by the entry chosen on the home page.
# (button key, title, one line, patient)
SITUATIONS = {
    "urgent": [
        ("svc_high_risk_delivery", "Labour has started",
         "She is in labour and needs to be admitted.", "A mother"),
        ("svc_mother_2", "The mother has an emergency",
         "A serious problem in pregnancy, during birth or after it, or she needs a transfer.",
         "A mother"),
        ("svc_nicu", "The newborn needs care",
         "A baby under 28 days old: born early, breathing trouble or another serious problem.",
         "A newborn"),
    ],
    "compare": [
        ("svc_high_risk_delivery", "Planning where to give birth",
         "Not in labour yet and choosing a hospital, also when a doctor has said the "
         "pregnancy is high-risk.", "A mother"),
        ("svc_nicu", "Newborn care, planned ahead",
         "For a baby under 28 days old: where newborn intensive care is, before it is needed.",
         "A newborn"),
    ],
}
SITUATIONS["area"] = SITUATIONS["compare"]


def pick_situation(label, title):
    """Button callback for step 2: remember the situation, then ask what is happening."""
    st.session_state["situation"] = title
    pick_patient(label)


def plan_area():
    """Button callback: a planner goes straight to the coverage map of an area."""
    st.session_state["situation"] = "Planning services for an area"
    if st.session_state.get("service_label") not in SERVICES:
        st.session_state["service_label"] = list(SERVICES.keys())[0]
    go("area")


def path_banner(role: str) -> None:
    """Top of steps 2 and 3: which of the three paths this is. Red only for the urgent one."""
    kind = "urgent" if role == "urgent" else "quiet"
    label = "Urgent" if role == "urgent" else "Not urgent"
    show_html(f'<div class="sa-path {kind}"><p class="k">{label}</p>'
              f'<p class="t">{esc(MODES[role])}</p><p class="g">{esc(PATHS[role]["get"])}</p></div>')


def screen_service() -> None:
    """Step 2: pick the service. The view was decided in step 1."""
    role = st.session_state.get("role", "urgent")
    st.session_state["last_screen"] = "service"
    top_bar("home")
    space(8)
    path_banner(role)

    with st.container():
        marker("cream")
        show_html(
            '<p class="sa-step">Step 2 of 4</p>'
            f'<h2 class="sa-h2">{esc(PATHS[role]["ask"])}</h2>'
            '<p class="sa-p" style="max-width:700px;margin:10px 0 14px 0">For mothers and newborns, we rate each '
            "facility's record on signals based on WHO" + src("who") + " and Government of "
            "India" + src("fru") + src("fbnc") + " standards.</p>"
        )
        cols = st.columns(3)
        for col, (key, title, line, label) in zip(cols, SITUATIONS[role]):
            service = SERVICES[label]
            with col.container():
                marker("card")
                # the icon is decoration (it shows whether this is about a mother or a baby);
                # text first, so the title lines up with the button; the icon sits top right
                show_html(f'<div class="sa-card-head"><div><p class="sa-card-title sa-sit-title">{esc(title)}</p>'
                          f'<p class="sa-small sa-sit-line">{esc(line)}</p></div>'
                          '<img class="sa-card-icon" alt="" '
                          f'src="data:image/png;base64,{CARD_ICON_BASE64[service]}"></div>')
                st.button("Choose", on_click=pick_situation, args=(label, title), key=key,
                          type="primary")
        if role != "urgent":
            with st.container():
                marker("links")
                st.button("I am planning services for an area: go to the coverage map",
                          on_click=plan_area, key="sit_planner")

        show_html('<hr class="sa-side-rule" style="margin:8px 0 22px 0"><h2 class="sa-h2" '
                  'style="font-size:22px">Coming soon</h2>'
                  '<p class="sa-small" style="margin-top:8px">Not available yet. These cards '
                  "are switched off on purpose.</p>")
        for start in range(0, len(COMING_SOON), 3):
            cols = st.columns(3)
            for col, name in zip(cols, COMING_SOON[start:start + 3]):
                with col.container():
                    marker("soon")
                    show_html(f'<p class="sa-soon-title">{esc(name)}</p>'
                              '<p class="sa-small" style="margin:8px 0 6px 0">This service is '
                              "not available yet.</p>")
                    st.button("Coming soon", disabled=True, key=f"soon_{name}")


def screen_needs() -> None:
    """Step 3: what is happening. Each answer becomes something to look for in the records."""
    role = st.session_state.get("role", "urgent")
    label = st.session_state.get("service_label")
    if label not in SERVICES:
        label = list(SERVICES.keys())[0]
        st.session_state["service_label"] = label
    service = SERVICES[label]
    st.session_state["last_screen"] = "needs"
    top_bar("home", back=(go, ("service",)))
    space(8)
    path_banner(role)

    main, _ = st.columns([1.6, 1])
    with main:
        who = "the baby" if service == "nicu" else "her"
        show_html(
            '<p class="sa-step">Step 3 of 4</p>'
            '<h1 class="sa-h1 small">'
            + ("What is happening?" if role == "urgent" else "What do you know so far?") + "</h1>"
            
            f'<p class="sa-p" style="margin-top:10px">Tick what applies to {who}. Each answer '
            "tells us what to look for in a facility's record. Leave everything empty if you "
            "are not sure.</p>"
            '<div class="sa-note" style="margin-top:12px">This is not a diagnosis. It only '
            "decides which words we look for in the records, and a record that does not "
            "mention something may still have it.</div>"
        )
        why = situation_rule(service)["why"]
        if why:
            source = situation_rule(service).get("src")
            show_html(f'<p class="sa-p" style="margin-top:14px"><b>{esc(st.session_state["situation"])}.'
                      f'</b> {esc(why)}{src(source) if source else ""}</p>')
        for heading, options in offered(service):
            show_html(f'<h3 class="sa-h3" style="margin-top:18px">{esc(heading)}</h3>')
            for key, text, need, want, note in options:
                looks = item_names(need + want)
                extra = (" In India, women with a high-risk pregnancy are linked to the "
                         "nearest First Referral Unit (Ministry of Health and Family Welfare)."
                         if key == "highrisk" else "")
                st.checkbox(text, key=f"ask_{service}_{key}",
                            help=(f"We will look for: {looks}.{extra}" if looks else note))
        told = answers(service)
        if told["needs"] or told["wants"]:
            line = []
            if told["needs"]:
                line.append(f"<b>We will look for:</b> {esc(item_names(told['needs']))}.")
            if told["wants"]:
                line.append(f" <b>Also shown:</b> {esc(item_names(told['wants']))}.")
            show_html(f'<p class="sa-p" style="margin-top:14px">{"".join(line)}</p>')
        space(6)
        st.button("Continue", on_click=needs_done, args=(role,), key="needs_continue", type="primary")


def results_heading(options) -> None:
    place = options["location"]["short"]
    title = SERVICE_TEXT[options["service"]]["title"]
    show_html(f'<h1 class="sa-h1 small">{esc(title)} near {esc(place)}</h1>')


def screen_where(mode: str, found: dict) -> None:
    """Step 3: the one thing still missing is where the user is starting from."""
    label = st.session_state.get("service_label")
    if label not in SERVICES:
        label = list(SERVICES.keys())[0]
        st.session_state["service_label"] = label
    top_bar("home", back=(go, ("needs",)))
    space(8)

    path_banner(mode)
    main, _ = st.columns([1.6, 1])
    with main:
        show_html(
            '<p class="sa-step">Step 4 of 4</p>'
            f'<h1 class="sa-h1 small">{esc(PATHS[mode]["where"])}</h1>'
            f'<p class="sa-p" style="margin-top:10px">'
            f'{esc(SERVICE_TEXT[SERVICES[label]]["title"])} '
            f'&middot; {esc(VIEW_LABELS[mode].lower())}</p>'
        )
        form_area = st.container()     # filled below, after the device location is handled
        problem_area = st.container()

        # The browser can supply the location. Handle its answer before the text box is
        # drawn, because a text box cannot be filled in after it is on the page.
        if GEO_BUTTON is not None:
            show_html('<p class="sa-small" style="margin-top:6px">or</p>')
            answer = GEO_BUTTON(key="geo_button", default=None)
            if isinstance(answer, dict) and answer.get("at") != st.session_state.get("geo_at"):
                st.session_state["geo_at"] = answer.get("at")
                if "lat" in answer and "lon" in answer:
                    text = f"{float(answer['lat']):.5f}, {float(answer['lon']):.5f}"
                    st.session_state["geo_query"] = text
                    st.session_state["place_query"] = text
                    st.session_state.pop("geo_error", None)
                    st.rerun()
                else:
                    st.session_state["geo_error"] = (
                        "Your browser did not share your location. Type a place instead.")
            if st.session_state.get("geo_error"):
                show_html(f'<div class="sa-note">{esc(st.session_state["geo_error"])}</div>')
            show_html('<p class="sa-fine">Your browser will ask for permission. The location '
                      "is used only to measure distances for this search.</p>")
        with st.container():
            marker("links")
            st.button("Change what you are looking for", on_click=go, args=("service",),
                      key="where_change")
            st.button("Clear the box", on_click=clear_place, key="where_clear")

        with form_area:
            with st.form("where_form", border=False):
                st.text_input(
                    "Place name, 6-digit PIN code, or latitude, longitude",
                    key="place_query", placeholder="For example: Araria",
                )
                show_html('<p class="sa-small">Examples: Araria &nbsp;&middot;&nbsp; 800001 '
                          "&nbsp;&middot;&nbsp; 25.61, 85.14</p>")
                st.form_submit_button(PATHS[mode]["go"], type="primary", on_click=place_done)

        with problem_area:
            if found["status"] == "error":
                show_html(f'<div class="sa-note">{esc(found["message"])}</div>')
            elif found["status"] == "choices":
                pick_among(found)
                st.button(PATHS[mode]["go"] + " for this place", on_click=confirm_place,
                          args=(found["term"],), key="where_confirm", type="primary")


def screen_results(mode: str) -> None:
    st.session_state["last_screen"] = mode
    st.session_state["role"] = mode
    try:
        found = resolve_place(st.session_state.get("place_query", ""))
    except Exception as exc:  # show the problem instead of a blank page
        found = {"status": "error", "message": f"The place search failed: {exc}"}

    if not place_is_settled(found) or st.session_state.get("edit_place"):
        screen_where(mode, found)
        return

    top_bar("home", back=(edit_place, ()))
    space(8)
    left, right = st.columns([2, 1], gap="large")
    with right.container():
        options = search_panel(mode, found)
    service = options["service"]

    with left:
        results_heading(options)
        if mode != "urgent":
            with st.container():
                marker("viewtabs")
                st.button("The list", on_click=go, args=("compare",), key="tab_list",
                          type="primary" if mode == "compare" else "secondary")
                st.button("The map", on_click=go, args=("area",), key="tab_map",
                          type="primary" if mode == "area" else "secondary")
        call_first(service)
        for note in answers(service)["notes"]:
            show_html(f'<div class="sa-note" style="margin-top:12px">{esc(note)}</div>')
        if mode == "urgent":
            view_urgent(options)
        elif mode == "compare":
            view_compare(options)
        else:
            view_area(options)


def plan_box(main: pd.DataFrame, plan: dict, service: str, other_first=None) -> None:
    """The suggestion in words: first choice, backup and what would change it.

    other_first is the first choice the model gives when records that list unusually many
    phone numbers are set aside (None when there is no such record in the plan, or nothing
    else to go to)."""
    first = main.iloc[plan["first"]]
    parts = ['<div class="sa-plan"><p class="t">Suggested plan</p>']
    in_plan = [main.iloc[i] for i in (plan["first"], plan["backup"]) if i >= 0]
    for row in [r for r in in_plan if mixed_record(r)]:
        asks = " ".join(esc(q) for q in phone_questions(row, service))
        parts.append(
            f'<p class="d"><b>Call {esc(row["name"])} before you set off.</b> Its record '
            f"{mixed_words(row)}. Ask: {asks}</p>")
    if any(mixed_record(r) for r in in_plan):
        if other_first is not None:
            parts.append(
                '<p class="d"><b>If they cannot confirm, or you cannot reach them:</b> go '
                f'first to {esc(other_first["name"])}, about {km(other_first["distance_km"])} '
                f'away. Its record: {esc(low(rating_label(other_first, service)))}. That is '
                "what this plan becomes when such records are set aside.</p>")
        else:
            parts.append('<p class="d">Setting such records aside leaves no other facility '
                         "to suggest here, so the plan below keeps them.</p>")
    if not plan["worth_it"]:
        parts.append(
            '<p class="d"><b>No option here is a clear choice.</b> Every facility on this list '
            "is far away, or its record says little, or both. Contact your district health "
            "office as well, and call before travelling. The least bad option is below.</p>")
    parts += [
             f'<p class="d"><b>Go first to {esc(first["name"])}</b>, about '
             f'{km(first["distance_km"])} away. Its record: '
             f'{esc(low(rating_label(first, service)))}.</p>']
    if plan["backup"] >= 0:
        backup = main.iloc[plan["backup"]]
        onward = plan["between"][plan["first"], plan["backup"]]
        parts.append(f'<p class="d"><b>If it cannot help, go on to {esc(backup["name"])}</b>, '
                     f'about {km(onward)} from the first choice '
                     f'({km(backup["distance_km"])} from where you are now). Its record: '
                     f'{esc(low(rating_label(backup, service)))}.</p>')
        if float(backup["distance_km"]) >= FAR_BACKUP_KM:
            parts.append('<p class="d"><b>The backup is far away.</b> Before travelling that '
                         "far, call the nearer facilities listed below: their records say "
                         "less, but one of them may be able to help.</p>")
    else:
        parts.append('<p class="d">No backup: no other facility on this list is worth a '
                     "second journey under the assumptions below.</p>")
    if plan["break_even"] is None and plan.get("nearest_overall") == first["unique_id"]:
        parts.append('<p class="d">The first choice is also the nearest option.</p>')
    elif plan["break_even"] is None:
        parts.append('<p class="d">Nearer facilities are listed below. They are not in the '
                     "suggestion because their record does not mention everything you asked "
                     "for, or their name points to another specialty.</p>")
    else:
        nearest = main.iloc[0]
        parts.append(
            f'<p class="d">The nearest option is {esc(nearest["name"])}, about '
            f'{km(nearest["distance_km"])} away. Going there first is the better plan only if '
            f'you believe it is at least <b>{plan["break_even"]:.0%}</b> likely to be able to '
            f'help today. We assumed {plan["p"][0]:.0%}, because its record reads: '
            f'{esc(low(rating_label(nearest, service)))}.</p>')
    parts.append('<p class="f">This is a suggestion, not a guarantee. Nobody has measured how '
                 "likely a facility is to help, so we use how much its record mentions as a "
                 "stand-in. Call before travelling.</p></div>")
    show_html("".join(parts))


def plan_workings(main: pd.DataFrame, plan: dict, service: str) -> None:
    """Everything behind the suggestion, for a reader who wants to check it."""
    if service == "nicu":
        stand_in = ("NICU with ventilator or specialist 75%, special newborn unit or CPAP 50%, "
                    "NICU named, no details 25%, basic newborn care only 0%.")
    else:
        m = len(COUNTED[service])
        stand_in = (f"the number of items its record mentions, divided by {m + 1}. A record "
                    f"that mentions all {m} gets {m / (m + 1):.0%}, never 100%, because a "
                    "record is a claim.")
    show_html(
        '<p class="sa-small"><b>What is compared.</b> For each facility: the distance to it, '
        "plus the extra travel you should expect if it turns out it cannot help. In that "
        "case you go on to the best backup. If the backup cannot help either, care is not "
        f"reached, and we count that as {PENALTY_KM} km of travel. The facility with the "
        "lowest total is the suggested first choice.</p>"
        '<p class="sa-small" style="margin-top:10px"><b>The chance a facility can help</b> is '
        f"not in the data. As a stand-in we use: {stand_in}</p>"
        '<p class="sa-small" style="margin-top:10px"><b>Left out.</b> Road distance and travel '
        "time (distances are straight lines), whether a bed or doctor is free today, "
        "facilities whose location is not confirmed, and facilities whose name points to "
        "another specialty (an eye hospital, a dental clinic).</p>"
    )
    order = np.argsort(plan["cost"], kind="stable")[:5]
    lines = ['<table class="sa-work"><caption>Lowest totals, best first (up to five)</caption>'
             "<tr><th>Go first to</th><th>Distance</th><th>Assumed chance</th>"
             "<th>Backup</th><th>Expected travel</th></tr>"]
    for j in order:
        row = main.iloc[j]
        b = plan["backups"][j]
        backup_name = esc(main.iloc[b]["name"]) if b >= 0 else "none"
        lines.append(
            f'<tr><td>{esc(row["name"])}</td><td>{km(row["distance_km"])}</td>'
            f'<td>{plan["p"][j]:.0%}<span class="nc">{esc(rating_label(row, service))}</span>'
            f'</td><td>{backup_name}</td><td><b>{km(plan["cost"][j])}</b></td></tr>')
    lines.append("</table>")
    show_html("".join(lines))

    other = []
    for penalty in (PENALTY_KM // 2, PENALTY_KM * 2):
        again = decide(main["distance_km"], plan["p"], plan["between"], penalty,
                       break_even=False)
        if again["first"] != plan["first"]:
            other.append(f"at {penalty:,} km it would be {esc(main.iloc[again['first']]['name'])}")
    if other:
        text = (f"The first choice depends on the {PENALTY_KM} km figure: "
                + " and ".join(other) + ".")
    else:
        text = (f"The first choice stays the same if not reaching care is counted as "
                f"{PENALTY_KM // 2:,} km or {PENALTY_KM * 2:,} km instead of {PENALTY_KM} km.")
    show_html(f'<p class="sa-small" style="margin-top:10px"><b>How firm is this?</b> {text}</p>')


def view_urgent(options) -> None:
    service, location = options["service"], options["location"]
    df = fetch(service, location, URGENT_RADIUS_KM, "distance", 500)
    if df is None:
        return
    main = df[df["list_section"] == "1_main"].sort_values("distance_km", kind="stable")
    main = main.reset_index(drop=True)
    uncertain = df[df["list_section"] != "1_main"]

    if main.empty and uncertain.empty:
        st.error(
            f"No facility whose record mentions {SERVICE_TEXT[service]['mentions']} was found within "
            f"{URGENT_RADIUS_KM} km. This does not prove none exists, only that none appears "
            "in the dataset. Contact your district health office."
        )
        return

    if not main.empty:
        doubtful = main["name"].map(lambda name: bool(name_points_elsewhere(name, service)))
        fit = main[~doubtful].reset_index(drop=True)       # the suggestion uses only these
        used, number = set(), 0
        needs = answers(service)["needs"]
        asked_for = ""
        if needs and not fit.empty:
            passing = fit[fit.apply(lambda row: meets_needs(row, needs), axis=1)]
            if passing.empty:
                asked_for = (f"<b>No facility within {URGENT_RADIUS_KM} km has a record that "
                             f"mentions all of: {esc(item_names(needs))}.</b> The suggestion "
                             "below does not take that into account. Ask about it first when "
                             "you call."
                             + (" If it cannot wait, call 112 or 108 for an ambulance."
                                if "emergency" in needs else ""))
            else:
                asked_for = (f"{len(passing)} of {len(fit)} facilities have a record that "
                             f"mentions all of: {esc(item_names(needs))}. The suggestion uses "
                             "only these. The others are still listed below.")
                fit = passing.reset_index(drop=True)
        if asked_for:
            show_html(f'<div class="sa-note" style="margin-top:14px">{asked_for}</div>')
        if fit.empty:
            show_html('<div class="sa-plan"><p class="t">No suggested plan</p><p class="d">'
                      "Every facility found has a name that points to another specialty, so "
                      "none is suggested. They are listed below. Contact your district health "
                      "office as well.</p></div>")
        else:
            p = np.array([chance(row, service) for _, row in fit.iterrows()])
            between = distances_between(fit["lat"], fit["lon"])
            plan = decide(fit["distance_km"], p, between)
            plan["between"] = between
            plan["nearest_overall"] = main.iloc[0]["unique_id"]
            # When the plan rests on a record that lists unusually many phone numbers, also
            # work out the plan without such records, so both can be shown side by side.
            other_first = None
            mixed = fit.apply(lambda row: bool(mixed_record(row)), axis=1)
            in_plan = [i for i in (plan["first"], plan["backup"]) if i >= 0]
            if any(mixed.iloc[i] for i in in_plan) and (~mixed).any():
                clean = fit[~mixed].reset_index(drop=True)
                p_clean = np.array([chance(row, service) for _, row in clean.iterrows()])
                other = decide(clean["distance_km"], p_clean,
                               distances_between(clean["lat"], clean["lon"]), break_even=False)
                other_first = clean.iloc[other["first"]]
            plan_box(fit, plan, service, other_first)
            if doubtful.any():
                show_html(f'<p class="sa-small" style="margin-top:8px">{int(doubtful.sum())} '
                          "facility(ies) were left out of the suggestion because their name "
                          "points to another specialty. They are still listed below.</p>")

            show_html('<h2 class="sa-h2 ruled">First choice</h2>')
            show_card(fit.iloc[plan["first"]], 1, service, prominent=True)
            used.add(fit.iloc[plan["first"]]["unique_id"])
            number = 1
            if plan["backup"] >= 0:
                show_html('<h2 class="sa-h2 ruled">Backup if the first choice cannot help</h2>')
                show_card(fit.iloc[plan["backup"]], 2, service, prominent=True)
                used.add(fit.iloc[plan["backup"]]["unique_id"])
                number = 2

        others = main[~main["unique_id"].isin(used)].head(3)
        if len(others):
            show_html(f'<h2 class="sa-h2 ruled">Other options within {URGENT_RADIUS_KM} km, '
                      "nearest first</h2>")
            for _, row in others.iterrows():
                number += 1
                show_card(row, number, service)

        if not fit.empty:
            with st.expander("How this suggestion was worked out"):
                plan_workings(fit, plan, service)
    else:
        st.warning(
            "No facility with a confirmed location was found. The options below mention the "
            "service, but their location is uncertain."
        )

    if not uncertain.empty:
        nearest_uncertain = uncertain.assign(
            closest=uncertain[["distance_km", "distance_km_by_coords"]].min(axis=1)
        ).sort_values("closest").head(3)
        if st.checkbox(
            f"Also show {len(nearest_uncertain)} possible option(s) whose location is not "
            "confirmed",
            key="urgent_show_uncertain",
        ):
            for i, (_, row) in enumerate(nearest_uncertain.iterrows(), start=1):
                show_card(row, i, service)


def one_page(df: pd.DataFrame, options, shown: int = 10) -> str:
    """A page to print or keep: the options at the top of the list and what each record says."""
    service, location = options["service"], options["location"]
    told = answers(service)
    situation = st.session_state.get("situation")
    head = [f"<h1>{esc(SERVICE_TEXT[service]['title'])} near {esc(location['short'])}</h1>",
            f"<p>Made with {BRAND} on {datetime.date.today().isoformat()}. Search area: "
            f"{int(options['radius'])} km around {esc(location['label'])}. The first "
            f"{min(shown, len(df))} of {len(df)} options on the list are shown.</p>"]
    if situation:
        head.append(f"<p><b>Situation:</b> {esc(situation)}</p>")
    if told["ticked"]:
        head.append("<p><b>What you told us:</b> " + "; ".join(esc(t) for t in told["ticked"]) + "</p>")
    head.append("<p class='warn'><b>Call before travelling.</b> \"Mentioned\" means the public "
                "record says it. Nobody has confirmed it, and records can be out of date. "
                "\"Not mentioned\" means the record is silent, not that the facility lacks it. "
                "Distances are straight lines. This is not medical advice.</p>")
    body = []
    for number, (_, row) in enumerate(df.head(shown).iterrows(), start=1):
        address = ", ".join(esc(part) for part in [row["address_line1"], row["city"], row["state"], row["pin"]]
                            if has(part))
        dist = (f"About {km(row['distance_km'])}, straight line" if row["list_section"] == "1_main"
                else "Location not confirmed: ask for the address")
        phone = esc(row["phone"]) if has(row["phone"]) else "none usable in the record"
        body.append(f"<section><h2>{number}. {esc(row['name'])}</h2>"
                    f"<p>{dist}{' &middot; ' + address if address else ''}<br><b>Phone: {phone}</b></p>"
                    + summary_html(row, service) + asked_html(row, service)
                    + "<p><b>Ask on the phone</b></p>" + questions_html(row, service, "") + "</section>")
    style = ("body{font-family:Arial,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;"
             "color:#222;font-size:14px;line-height:1.45}h1{font-size:22px}h2{font-size:16px;"
             "margin:0 0 4px 0}section{border-top:1px solid #999;padding:10px 0;"
             "page-break-inside:avoid}p{margin:4px 0}ol{margin:2px 0 0 20px;padding:0}"
             ".warn{border:1px solid #999;padding:8px}")
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{BRAND} summary</title><style>{style}</style></head><body>"
            + "".join(head) + "".join(body) + "</body></html>")


def view_compare(options) -> None:
    service, location, radius_km = options["service"], options["location"], options["radius"]
    sort = "evidence" if options["by_rating"] else "distance"
    filtered = (options["only_24h"] or options.get("only_strong") or options.get("only_public")
                or options.get("by_match"))
    # with a filter on, look through more records so that the 30 shown are the best 30 left
    df = fetch(service, location, radius_km, sort, 200 if filtered else 30)
    if df is None:
        return
    if df.empty:
        st.warning(
            f"No facility whose record mentions {SERVICE_TEXT[service]['mentions']} was found within "
            f"{radius_km} km of {location['label']}. Try a larger distance. This does not "
            "prove none exists, only that none appears in the dataset."
        )
        return
    if options["by_rating"]:
        df = by_rating(df, service)
    elif options.get("by_match"):
        df = by_match(df, service)
        asked = answers(service)
        n_items = len(asked["needs"] + asked["wants"])
        show_html(f'<p class="sa-small" style="margin-top:10px">Ordered by how many of the '
                  f"{n_items} thing(s) you ticked each record mentions, then by distance. "
                  "A record that is silent about something is not a facility that lacks "
                  "it.</p>")

    total = len(df)
    if options["only_24h"]:
        df = df[df["hours_level"].map(counts_24h)]
        if df.empty:
            st.warning(
                f"None of the {total} option(s) has a record that mentions 24-hour emergency "
                "care or 24-hour service. Untick the box to see them all."
            )
            return
    if options.get("only_strong"):
        kept = df[df.apply(lambda row: well_rated(row, service), axis=1)]
        if kept.empty:
            what = ("detail about the newborn unit" if service == "nicu"
                    else "at least 3 of the 4 items")
            st.warning(
                f"None of the {len(df)} option(s) within {int(radius_km)} km has a record "
                f"with {what}. Untick the box to see them all. A weaker record does not "
                "mean the facility cannot help: it means the record says less."
            )
            return
        show_html(f'<div class="sa-note" style="margin-top:14px">{len(kept)} of {len(df)} '
                  "records are shown. The others say less, which is not proof that those "
                  "facilities offer less.</div>")
        df = kept
    if options.get("only_public"):
        types = operator_types(tuple(df["unique_id"]))
        if types is None:
            st.warning("The dataset's label for who runs a facility could not be read, so "
                       "this box has no effect. All options are shown.")
        else:
            public = df[df["unique_id"].map(lambda uid: types.get(uid) in PUBLIC_WORDS)]
            if public.empty:
                st.warning(
                    f"None of the {len(df)} option(s) within {int(radius_km)} km is listed "
                    "as public or government. Untick the box to see them all. Fewer than 5 "
                    "in 100 facilities in the dataset carry that label, which describes the "
                    "dataset, not necessarily what exists.")
                return
            show_html(f'<div class="sa-note" style="margin-top:14px">{len(public)} of '
                      f"{len(df)} are listed as public or government. Fewer than 5 in 100 "
                      "facilities in the dataset carry that label, which describes the "
                      "dataset, not necessarily what exists.</div>")
            df = public
    if filtered:
        df = df.head(30).reset_index(drop=True)

    show_html(f'<h2 class="sa-h2 ruled">Within {int(radius_km)} km: {len(df)} shown</h2>'
              '<p class="sa-small" style="margin-top:8px">Check what the record says and the '
              "phone first. The number of options matters less than whether one of them can "
              "take the patient. Up to 30 are listed.</p>")
    table = results_table(df, service)
    show_table(table)
    st.download_button(
        "Download this list (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name=f"{service}_options.csv", mime="text/csv", key="compare_download",
    )
    st.download_button(
        "Download a one-page summary to print", one_page(df, options).encode("utf-8"),
        file_name=f"{service}_summary.html", mime="text/html", key="compare_summary",
        help="A web page with the first 10 options and what to ask each one. Open it and "
             "print it from your browser.",
    )

    main = df[df["list_section"] == "1_main"]
    uncertain = df[df["list_section"] != "1_main"]
    number = 0
    for _, row in main.iterrows():
        number += 1
        show_card(row, number, service)
    if not uncertain.empty:
        show_html('<h2 class="sa-h2 ruled">Location not confirmed</h2>'
                  '<p class="sa-small" style="margin-top:8px">These records give two '
                  "conflicting locations. They are always listed last.</p>")
        for _, row in uncertain.iterrows():
            number += 1
            show_card(row, number, service)


def map_zoom(lats, lons, width_px: int = 620, height_px: int = 480) -> float:
    """Zoom level at which all the points fit on the map, with some margin.

    The map library draws the world 512 pixels wide at zoom 0 and doubles it at each level.
    """
    import math
    lat_mid = (max(lats) + min(lats)) / 2
    world_km = 40075.0 * math.cos(math.radians(lat_mid))
    wide_km = max((max(lons) - min(lons)) * 111.32 * math.cos(math.radians(lat_mid)), 0.5)
    tall_km = max((max(lats) - min(lats)) * 111.32, 0.5)
    margin = 1.35
    fit_wide = math.log2(world_km * width_px / (512 * wide_km * margin))
    fit_tall = math.log2(world_km * height_px / (512 * tall_km * margin))
    return max(3.0, min(15.0, fit_wide, fit_tall))


def draw_map(shown: pd.DataFrame, location: dict, service: str, number_of: dict) -> None:
    """Dots with the table's row numbers beside them and the facility's name on hover."""
    def rgba(colour: str, alpha: int) -> list:
        return [int(colour[i:i + 2], 16) for i in (1, 3, 5)] + [alpha]

    tier_colour = {"1_strong": rgba(PLUM, 235), "2_medium": rgba(YELLOW, 235)}
    places = []
    for _, row in shown.iterrows():
        n = number_of.get(row["unique_id"], "")
        places.append({
            "lat": float(row["lat"]), "lon": float(row["lon"]), "label": str(n),
            "colour": tier_colour.get(row["evidence_tier"], rgba("#8A7F6A", 235)),
            # shown in the hover box, so escape anything that could be read as HTML
            "title": f"{n}. {html.escape(str(row['name']))}",
            "detail": f"{html.escape(rating_label(row, service))} &middot; about {km(row['distance_km'])}",
        })
    start = [{"lat": location["lat"], "lon": location["lon"], "label": "", "colour": rgba(INK, 255),
              "title": "Your starting place", "detail": html.escape(str(location["label"]))}]
    lats = [p["lat"] for p in places + start]
    lons = [p["lon"] for p in places + start]
    try:
        import pydeck
        layers = [
            pydeck.Layer("ScatterplotLayer", data=places, get_position="[lon, lat]",
                         get_fill_color="colour", get_line_color=[255, 255, 255], stroked=True,
                         line_width_min_pixels=1.5, get_radius=60, radius_min_pixels=7,
                         radius_max_pixels=11, pickable=True),
            pydeck.Layer("ScatterplotLayer", data=start, get_position="[lon, lat]",
                         get_fill_color="colour", get_line_color=[255, 255, 255], stroked=True,
                         line_width_min_pixels=2, get_radius=90, radius_min_pixels=10,
                         radius_max_pixels=14, pickable=True),
            pydeck.Layer("TextLayer", data=places, get_position="[lon, lat]", get_text="label",
                         get_size=14, get_color=rgba(INK, 255), get_pixel_offset=[13, 0],
                         get_text_anchor="'start'", get_alignment_baseline="'center'",
                         font_weight=700, outline_width=3, outline_color=[255, 255, 255, 255],
                         font_settings={"sdf": True}),
        ]
        view = pydeck.ViewState(latitude=(max(lats) + min(lats)) / 2,
                                longitude=(max(lons) + min(lons)) / 2,
                                zoom=map_zoom(lats, lons), pitch=0)
        tooltip = {"html": "<b>{title}</b><br>{detail}",
                   "style": {"backgroundColor": "#FFFFFF", "color": INK, "fontSize": "14px",
                             "border": f"1px solid {RULE}", "padding": "8px 10px"}}
        st.pydeck_chart(pydeck.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                                    map_style="light"))
    except Exception:   # plain dots are better than no map
        frame = pd.DataFrame([{"lat": p["lat"], "lon": p["lon"],
                               "color": "#%02x%02x%02x" % tuple(p["colour"][:3]), "size": 200}
                              for p in places + start])
        st.map(frame, latitude="lat", longitude="lon", color="color", size="size")


def view_area(options) -> None:
    service, location, radius_km = options["service"], options["location"], options["radius"]
    df = fetch(service, location, radius_km, "distance", 500)
    if df is None:
        return
    if df.empty:
        st.error(
            f"No facility whose record mentions {SERVICE_TEXT[service]['mentions']} appears in the dataset "
            f"within {radius_km} km of {location['label']}. This is a possible coverage gap, "
            "but the dataset is incomplete, so check local sources."
        )
        return

    main = df[df["list_section"] == "1_main"].sort_values("distance_km")
    uncertain = df[df["list_section"] != "1_main"]
    best = main[main["evidence_tier"] == "1_strong"]
    open24 = main[main["hours_level"].map(counts_24h)]

    # Lead with the answer a planner can act on, not with counts.
    if best.empty:
        show_html(f'<div class="sa-note"><b>Possible gap:</b> no facility within '
                  f"{int(radius_km)} km has a record that {top_phrase(service)}.</div>")
    else:
        show_html('<p class="sa-p lg">' + summary_sentence(
            f"The nearest facility whose record {top_phrase(service)}", best.iloc[0],
            service) + "</p>")
    if open24.empty:
        show_html(f'<div class="sa-note"><b>Possible gap:</b> no facility within '
                  f"{int(radius_km)} km has a record that mentions 24-hour emergency care."
                  "</div>")

    show_html(f'<h2 class="sa-h2 ruled">Within {int(radius_km)} km</h2>')
    # Facilities cluster in towns, so on a map of the whole area the dots pile up. Say so in
    # words, keep the dots small and see-through, and offer a closer view of the nearest half.
    shown = main
    zoomed = False
    if len(main) >= 4:
        half_km = float(main["distance_km"].median())
        show_html(f'<p class="sa-p" style="margin-top:10px">Half of the {len(main)} facilities '
                  f"on the map are within {km(half_km)} of your starting place, so their dots "
                  "overlap. Zoom in with the + button, or tick the box.</p>")
        zoomed = st.checkbox("Zoom in on the nearest half", key="area_zoom")
        if zoomed:
            shown = main[main["distance_km"] <= half_km]
    number_of = {uid: i for i, uid in enumerate(df["unique_id"], start=1)}   # as in the table
    draw_map(shown, location, service, number_of)
    if zoomed:
        st.caption(f"Showing the {len(shown)} nearest of {len(main)} facilities. The other "
                   f"{len(main) - len(shown)} are farther than {km(half_km)} away and are in "
                   "the table below.")
    if service == "nicu":
        legend = ("Plum: NICU with ventilator or specialist. Yellow: special newborn unit or CPAP. "
                  "Grey: NICU named with no details, or basic care only.")
    else:
        n = len(COUNTED[service])
        legend = (f"Plum: all {n} mentioned. Yellow: {n - 1} of {n} mentioned. "
                  "Grey: fewer.")
    st.caption(
        f"Dark dot: your starting place. {legend} Point at a dot to see its name. Each number "
        "is the facility's row in the table below; pick that number under the map to open "
        "its full record. Facilities whose location is not confirmed are left off the map "
        "and listed in the table."
    )

    # The map itself cannot open a record (this Streamlit version has no click on map
    # dots), so the number seen on the map is picked here and the usual box appears.
    numbered = df.reset_index(drop=True)
    choices = [None] + list(range(len(numbered)))
    picked = st.selectbox(
        "See one facility's record: pick its number from the map", choices, key="area_pick",
        format_func=lambda i: "Choose a facility" if i is None
        else f"{i + 1}. {numbered.iloc[i]['name']}",
    )
    if picked is not None:
        show_card(numbered.iloc[picked], picked + 1, service)

    cols = st.columns(4)
    best_name = top_label(service)
    cols[0].metric("Ventilator/specialist" if service == "nicu"
                   else best_name[:1].upper() + best_name[1:], len(best))
    cols[1].metric("24-hour emergency", len(open24))
    cols[2].metric("Have a phone number", int(main["phone"].map(has).sum()))
    cols[3].metric("Location not confirmed", len(uncertain))
    ratings = main.apply(lambda row: rating_label(row, service), axis=1).value_counts()
    order = ([NICU_LEVELS[t][0] for t in ("1_strong", "2_medium", "3_weak", "4_basic")]
             if service == "nicu"
             else [f"{n} of {len(COUNTED[service])} mentioned"
                   for n in range(len(COUNTED[service]), -1, -1)])
    listed = [f"{esc(label)}: {int(ratings[label])}" for label in order if label in ratings]
    listed += [f"{esc(label)}: {int(n)}" for label, n in ratings.items() if label not in order]
    show_html(f'<p class="sa-small" style="margin-top:6px"><b>The {len(main)} facilities on '
              f'the map, by what the record says.</b> {"; ".join(listed)}.</p>')
    st.caption(
        f"Out of {len(df)} facilities in the dataset whose record mentions "
        f"{SERVICE_TEXT[service]['mentions']} within {int(radius_km)} km. This is not a census: a "
        "facility "
        "missing here may still exist."
    )

    table = results_table(df, service)
    show_table(table)
    st.download_button(
        "Download this list (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name=f"{service}_coverage.csv", mime="text/csv", key="area_download",
    )
    st.caption("Up to 500 facilities are shown.")


def genie_table(client, conversation_id, message_id, attachment_id) -> pd.DataFrame:
    """The rows behind one Genie answer.

    Read from the raw reply, not through the SDK's classes: Genie sends the rows as
    "data_typed_array" (one {"values": [{"str": ...}]} per row), which the SDK's classes
    drop, so they would come back empty. The plain "data_array" form is accepted too.
    """
    base = (f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}"
            f"/messages/{message_id}")
    if attachment_id:
        reply = client.api_client.do("GET", f"{base}/attachments/{attachment_id}/query-result")
    else:  # no attachment id to be had: the older address returns the message's query result
        reply = client.api_client.do("GET", f"{base}/query-result")
    statement = (reply or {}).get("statement_response") or {}
    manifest = statement.get("manifest") or {}
    columns = [column.get("name") for column in (manifest.get("schema") or {}).get("columns", [])]
    result = statement.get("result") or {}
    rows = result.get("data_array")
    if rows is None:
        rows = []
        for row in result.get("data_typed_array") or []:
            cells = row.get("values", []) if isinstance(row, dict) else row
            rows.append([next(iter(cell.values()), None) if isinstance(cell, dict) else cell
                         for cell in cells])
    expected = manifest.get("total_row_count", result.get("row_count"))
    if not rows and expected:
        raise ValueError(f"the query returned {expected} row(s) in a form this page cannot "
                         f"read ({', '.join(sorted(result)) or 'no result block'})")
    if result.get("external_links"):
        raise ValueError("the result is too large to show here")
    return pd.DataFrame(rows, columns=columns)


def ask_genie(question: str) -> dict:
    """Send one question to the Genie agent and collect its words, its queries and the rows.

    Returns {"text": [...], "queries": [{"title", "sql", "table"}], "problem": str or None}.
    Nothing is added or reworded here: the page shows what Genie returned.
    """
    import datetime
    from databricks.sdk import WorkspaceClient

    answer = {"question": question, "text": [], "queries": [], "problem": None}
    client = WorkspaceClient()
    try:
        message = client.genie.start_conversation_and_wait(
            GENIE_SPACE_ID, question, timeout=datetime.timedelta(minutes=GENIE_WAIT_MINUTES))
    except TimeoutError:
        answer["problem"] = (f"Genie did not answer within {GENIE_WAIT_MINUTES} minutes. "
                             "Try again, or ask a simpler question.")
        return answer
    except Exception as exc:  # the SDK raises when Genie reports that it failed
        answer["problem"] = f"Genie could not answer: {exc}"
        return answer

    status = getattr(getattr(message, "status", None), "value", None)
    if status and status != "COMPLETED":
        detail = getattr(getattr(message, "error", None), "error", None)
        answer["problem"] = f"Genie could not answer (status {status})." + (
            f" {detail}" if detail else "")
    message_id = getattr(message, "message_id", None) or getattr(message, "id", None)
    for part in getattr(message, "attachments", None) or []:
        text = getattr(part, "text", None)
        if text is not None and getattr(text, "content", None):
            answer["text"].append(text.content)
        query = getattr(part, "query", None)
        if query is None:
            continue
        found = {"title": getattr(query, "title", None) or getattr(query, "description", None),
                 "sql": getattr(query, "query", None), "table": None, "problem": None}
        try:
            # the id sits in different places depending on the SDK version installed
            attachment_id = (getattr(part, "attachment_id", None) or getattr(part, "id", None)
                             or getattr(query, "id", None))
            found["table"] = genie_table(client, message.conversation_id, message_id,
                                         attachment_id)
        except Exception as exc:  # the words and the query are still worth showing
            found["problem"] = f"The rows could not be loaded: {exc}"
        answer["queries"].append(found)
    if not answer["text"] and not answer["queries"] and not answer["problem"]:
        answer["problem"] = "Genie returned an empty answer. Try rewording the question."
    return answer


def floating_ask() -> None:
    """A button pinned to the bottom right corner that opens the questions page."""
    with st.container():
        marker("fab")
        st.button("Ask Genie about the data", on_click=go, args=("ask",), key="ask_fab")


def set_question(text: str) -> None:
    """Button callback: put an example into the question box."""
    st.session_state["genie_question"] = text


def screen_ask() -> None:
    """For planners: free questions about the data, answered by Databricks Genie."""
    if st.session_state.get("last_screen") != "ask":       # where the user came from
        st.session_state["ask_from"] = st.session_state.get("last_screen")
    st.session_state["last_screen"] = "ask"
    top_bar("home", back=(go, (st.session_state.get("ask_from"),)))
    space(8)
    main, _ = st.columns([2, 1])
    with main:
        show_html(
            '<h1 class="sa-h1 small">Ask a question about the data</h1>'
            '<p class="sa-p" style="margin-top:10px">For planning and analysis: counts and '
            "lists across India. The answer is written by Databricks Genie, an AI, which "
            "turns your question into a database query. The query and the rows it returned "
            "are shown, so you can check the answer.</p>"
            '<div class="sa-call" style="margin-top:16px"><p class="t">Not for choosing where '
            'to send a patient.</p><p class="d">Use "I need care for a patient now" on the '
            "home page for that.</p></div>"
            '<div class="sa-note" style="margin-top:14px">Every number counts facilities '
            "<b>whose record mentions</b> a service, not facilities confirmed to offer it. "
            "A facility missing from an answer may still exist.</div>"
        )
        if not GENIE_SPACE_ID:
            show_html('<div class="sa-note" style="margin-top:14px">This page is not set up: '
                      "the app has no Genie resource. Add one with the key 'genie-space' in "
                      "the app settings and redeploy.</div>")
            return

        with st.form("genie_form", border=False):
            st.text_input("Your question", key="genie_question",
                          placeholder="For example: " + GENIE_EXAMPLES[0])
            asked = st.form_submit_button("Ask", type="primary")
        show_html('<p class="sa-small" style="margin-top:4px">Or try one of these. Their '
                  "numbers were checked against our own queries.</p>")
        with st.container():
            marker("links")
            st.button("Clear the box", on_click=clear_question, key="genie_clear")
            for i, example in enumerate(GENIE_EXAMPLES):
                st.button(example, on_click=set_question, args=(example,),
                          key=f"genie_example_{i}")

        question = " ".join(str(st.session_state.get("genie_question", "")).split())
        if asked and len(question) < 8:
            show_html('<div class="sa-note">Type a full question first.</div>')
        elif asked:
            try:
                with st.spinner("Genie is working on it. This can take up to a minute..."):
                    st.session_state["genie_answer"] = ask_genie(question)
            except Exception as exc:  # show the problem instead of a blank page
                st.session_state["genie_answer"] = {
                    "question": question, "text": [], "queries": [],
                    "problem": f"The question could not be sent to Genie: {exc}"}

        answer = st.session_state.get("genie_answer")
        if not answer:
            return
        show_html(f'<h2 class="sa-h2 ruled">Answer</h2>'
                  f'<p class="sa-small" style="margin-top:8px">You asked: '
                  f'{esc(answer["question"])}</p>')
        if answer["problem"]:
            show_html(f'<div class="sa-note">{esc(answer["problem"])}</div>')
        for text in answer["text"]:
            with st.container():
                marker("prose")
                st.markdown(text)        # Genie's own words, as plain Markdown (no HTML)
        for number, found in enumerate(answer["queries"], start=1):
            if found["title"]:
                show_html(f'<p class="sa-small" style="margin-top:10px"><b>'
                          f'{esc(found["title"])}</b></p>')
            if found["problem"]:
                show_html(f'<div class="sa-note">{esc(found["problem"])}</div>')
            elif found["table"] is not None:
                show_table(found["table"])
                st.caption(f"{len(found['table'])} row(s) returned by the query.")
            if found["sql"]:
                with st.expander("The query Genie ran"):
                    st.code(found["sql"], language="sql")
        show_html('<p class="sa-fine" style="margin-top:14px">Written by an AI. Check the '
                  "words against the rows and the query before you rely on them.</p>")


ABOUT_SHORT = """
**What this tool does.** It shows what each facility's public record says about care for
mothers and newborns, and what it leaves out. It does not promise that a suggestion is right.

**Where the information comes from.** Records were put together automatically from web
pages. Every statement is a claim. Nobody has confirmed it.

**"Mentioned" and "Not mentioned".** "Mentioned" means the record says it. "Not mentioned"
means the record is silent. The facility may still have it.

**How the first choice is picked.** The nearest facility is not always the best first stop:
if it cannot help, the journey is lost. We weigh distance against how much the record says.

**A warning on a box** means the record may be wrong: a name that does not fit, too many
phone numbers, or an address that does not match the map.

**What the tool cannot know.** Whether a bed is free, whether a doctor is on duty, or how
long the road takes.

**Always call before you travel.** Each box tells you what to ask.
"""

# Sections for technical readers: (title, markdown). "@R@", "@B@" and so on are replaced by
# the constants used in the code, so the page cannot drift from what the tool does.
ABOUT_TECH = [
    ("Data and which facilities are listed", """
- **Source.** The Virtue Foundation dataset: about 10,000 healthcare facilities in India.
  Each record was assembled automatically from web pages.
- **Where we look.** Four parts of a record: services listed, procedures listed, equipment
  listed and the description. A facility is listed only if at least one of them mentions the
  service. A sentence repeated in several parts is counted once.
- **Sentences left out.** Wording that says a service is planned, under construction,
  coming soon or not offered. "Plans to establish a NICU" does not count. These are simple
  wording rules, so a few may slip through.
- **Facilities left out by name.** A name that points to an eye, dental, skin or homeopathy
  practice, or to a diagnostic centre that is not a hospital, is removed from both lists,
  unless the name also carries a word such as maternity, women, child, general or multi.
- **Scope.** Mother and newborn care only.
"""),
    ("How a record is rated", """
**A mother (birth emergency care).** Count how many of four things the record mentions:

| Item | Counts when the record mentions |
|---|---|
| C-section | a caesarean section |
| Blood | a blood bank or transfusion |
| 24-hour emergency care | "24/7", "24 hours" or "round the clock" next to emergency, casualty or trauma, or in the same clause as the service |
| Newborn care | a NICU, a special newborn care unit, a stabilisation unit or another neonatal service |

A result reads "3 of 4 mentioned". A facility is on the list if its record mentions
C-section or says it handles high-risk pregnancy. "Open 24 hours" alone is shown but not
counted, and neither is "within 24 hours".

**A newborn (newborn intensive care).** Four groups, by what the record says about the unit:

| Group | The record mentions | Indian guideline level |
|---|---|---|
| NICU with ventilator or specialist | neonatal ventilation or surfactant, or a NICU together with a neonatologist | III |
| Special newborn unit or CPAP | a special newborn care unit or neonatal CPAP | II |
| NICU named, no details | a NICU, with no equipment or staff detail | cannot be told |
| Basic newborn care only | a radiant warmer, newborn phototherapy or a stabilisation unit, without a NICU | I |

Ventilation counts only when written next to the newborn unit ("NICU with ventilators"),
not as one item in a list ("ICU, NICU, ventilator").

**Rows marked "for information, not counted".** "Handles high-risk pregnancy" is a general
claim, and "operating theatre" or "anaesthesia" may refer to any surgery. They are shown but
do not raise the rating.

**Basis.** The [WHO description of comprehensive emergency obstetric care](https://www.who.int/publications-detail-redirect/9789241547734),
the [Government of India standard for First Referral Units](https://pmc.ncbi.nlm.nih.gov/articles/5144115),
the [Government of India Facility Based Newborn Care guidelines](https://nhm.gov.in/images/pdf/programmes/CH-Programmes/FBNC/FBNC%20Operational%20Guidelines%202025.pdf)
and the [levels used by the National Neonatology Forum of India](https://www.nnfi.org/assests/upload/download-pdf/FINAL_FORM_LEVEL_IIIB_2022-23.pdf).
All sources are listed under "Sources" below. We worked from
summaries of these documents, not the full texts. The groups are clues from wording, not
official gradings.
"""),
    ("The decision model behind the suggested first choice", None),      # written with formulas
    ("Location", """
- Each record has a map position and a postal code. Within 50 km of each other: confirmed.
- When they disagree, we keep the one that matches other facilities in the same city, and
  the box says "Check the address".
- When that does not settle it: "location not confirmed". The facility is listed last, shown
  with both possible distances, and left out of the decision model.
- Your starting point is matched to the India Post directory and measured from the centre of
  that area. Coordinates can be entered directly.
- All distances are straight lines.
"""),
    ("Warnings and their thresholds", """
| Warning on the box | Shown when | Can it be first choice or backup? |
|---|---|---|
| Check this one | the name points to another specialty (heart, cancer, kidney, orthopaedic), or to a children's hospital when the patient is a mother | No |
| Check this record | the record lists @P@ or more phone numbers (about 3 in 100 records do) | Yes. The plan then tells you to call first, and shows what the plan becomes without such records |
| Check the address | postal code and map position disagree | Yes |
| "uncomplicated" C-section | the C-section wording carries "uncomplicated", "elective" or "planned" | Yes |

- The number of web pages is not used on the box: the dataset keeps at most 50 per record
  and more than a third of records reach 30. The popup still notes 30 or more pages or 10 or
  more phone numbers.
- The plan says the backup is far when it is @F@ km or more from the starting point.
- Many phone numbers are not proof of a mixed record. Of the 32 facilities on the two lists
  with @P@ or more, 11 give 250 beds or more (medical colleges and large hospital groups),
  where so many numbers are to be expected. Others give 10 to 30 beds. So such records are
  not removed from the plan: the box shows both numbers and the plan says to call first.
- A mixed record whose name looks ordinary is not caught.
- The thresholds are our choices.
"""),
    ("Situations and the questions about what is happening", """
- The situation chosen at step 2 decides which list is used (mothers or newborns) and which
  questions follow. "The mother has an emergency" requires a record that mentions 24-hour
  emergency care. When planning a birth, ticking "A doctor has said the pregnancy is
  high-risk" looks for C-section.
- Each answer ticked becomes something to look for: "heavy bleeding" looks for blood and
  24-hour emergency care.
- When some facilities mention everything asked for, the first choice and the backup come
  only from those. When none does, the page says so and gives the usual suggestion.
- The questions to ask on the phone are built from what the record does not mention.
- **Not checked against a clinical source.** The links from a situation or a sign to what we
  look for were written by the team. They are a way to search records, not a diagnosis.
"""),
    ("Other parts of the tool", """
- **Other lists.** In the full list you choose the order: most mentioned first, nearest
  first, or, when you ticked something, best match to what you told us (the records that
  mention more of the ticked things come first). Preferences when planning a birth (normal
  delivery, pain relief in labour, checkups, care after the birth, vaccination, ultrasound,
  blood tests) are shown on each box and never used to leave a facility out: few records
  mention them (pain relief in labour: about 100 of the 660 records on the mother list).
  We also looked at whether a record lists paediatrics among its specialties, as a hint
  for vaccination. It is not shown: 468 of the 660 records on the mother list and 677 of
  the 757 on the newborn list carry that tag, so it tells facilities apart very little, and
  a specialty tag is not counted as evidence anywhere in this tool. Size (beds, doctors) is shown as the record gives it and is never used to order or
  filter. The list can be narrowed to facilities the dataset labels public or government;
  few carry that label (fewer than 5 in 100 in the whole dataset), and there is no label for
  NGO or trust hospitals.
- **Card for the driver.** Two sentences in the main language of the facility's state, the
  name, address and phone as the record gives them, a map link and three emergency numbers.
  Nine Indian languages and English. The sentences were written for this tool and have not
  been checked by native speakers. Which language a state gets is our choice.
- **Questions about the data.** Passed to Databricks Genie, which writes a query over the
  same tables. The page shows its words, the rows and the query. Three example questions were
  checked against our own queries. Others were not, so read the rows and the query. Genie is
  told to cover mother and newborn care only and never to recommend a facility for a patient.
"""),
    ("Sources", """
- WHO, *Monitoring emergency obstetric care: a handbook* (2009). Comprehensive emergency obstetric care: the basic functions plus caesarean section and blood transfusion. https://www.who.int/publications-detail-redirect/9789241547734
- First Referral Units in India: caesarean section, newborn care and blood storage, available 24 hours. https://pmc.ncbi.nlm.nih.gov/articles/5144115
- Ministry of Health and Family Welfare, Government of India, on PMSMA: women with a high-risk pregnancy are linked to the nearest First Referral Unit. https://mohfw.gov.in/press-info/7478
- National Health Mission, *Facility Based Newborn Care Operational Guidelines 2025*: newborn stabilisation units and special newborn care units. https://nhm.gov.in/images/pdf/programmes/CH-Programmes/FBNC/FBNC%20Operational%20Guidelines%202025.pdf
- National Neonatology Forum of India, accreditation form for level IIIB units (2022-23): levels of newborn care by type of breathing support. https://www.nnfi.org/assests/upload/download-pdf/FINAL_FORM_LEVEL_IIIB_2022-23.pdf
- Emergency numbers 112, 102 and 108: the india.gov.in helpline directory and a Ministry of Health and Family Welfare reply in Parliament (no link kept).
- Databricks Free Edition limitations. https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
- Data: the Virtue Foundation dataset of healthcare facilities in India, provided for the challenge. Challenge brief: https://docs.google.com/document/d/e/2PACX-1vQxhDNQmgZhPgjG0M6LCb_UXNV8CJryqBNQQuqWfIg_dm40ZU1AYAXzZKViC6haFoCmLNiRVfUSBMjf/pub

We worked from summaries of the guideline documents, not the full texts.
"""),
    ("What this tool cannot tell you", """
- Whether a bed is free or a doctor is on duty right now.
- Road distance or travel time.
- Whether a record is correct. Some mix details from more than one facility.
- Whether a phone number or website is current, or belongs to this facility.
- Anything outside mother and newborn care.
- Whether a facility missing from the results offers the service. It may.

Always call the facility to confirm before sending a patient.
"""),
]

# The decision model, as (kind, text): "md" is markdown, "tex" is a displayed formula.
ABOUT_MODEL = [
    ("md", """
When care is needed now, the nearest facility is not always the best first stop: if it
cannot help, the journey there is lost. The model picks the first stop that minimises the
expected distance travelled, counting a complete failure as a fixed penalty.

**Notation**

| Symbol | Meaning |
|---|---|
| $J$ | facilities with a confirmed location within @R@ km whose name passes the name rule. When at least one of them mentions everything the answers require, only those |
| $d_j$ | straight-line distance from the starting point to facility $j$ |
| $d_{jl}$ | straight-line distance from facility $j$ to facility $l$ |
| $p_j$ | stand-in for the chance that $j$ can help. **An assumption, not a measured probability** |
| $B$ | penalty when neither stop can help: @B@ km |

**Stand-in $p_j$.** For a mother, with $n_j$ of $m = 4$ items mentioned:
"""),
    ("tex", r"p_j = \frac{n_j}{m + 1} \in \{0,\ 0.2,\ 0.4,\ 0.6,\ 0.8\}"),
    ("md", """
so a record that mentions all four gets 0.8, never 1. For a newborn, by group: 0.75
(NICU with ventilator or specialist), 0.50 (special newborn unit or CPAP), 0.25 (NICU
named, no details), 0 (basic newborn care only).

**Cost after a failed first stop.** Go on to the best backup, or stop if that is no better
than the penalty:
"""),
    ("tex", r"F_j = \min\Big\{\, B,\ \min_{l \in J,\ l \neq j} \big[\, d_{jl} + (1 - p_l)\,B \,\big] \Big\}"),
    ("md", "**Expected cost of going to $j$ first.**"),
    ("tex", r"C_j = d_j + (1 - p_j)\,F_j"),
    ("md", "**Suggested first choice, and its backup.**"),
    ("tex", r"j^{*} = \arg\min_{j \in J} C_j \qquad l^{*} = \arg\min_{l \neq j^{*}} \big[\, d_{j^{*}l} + (1 - p_l)\,B \,\big]"),
    ("md", """
The backup $l^{*}$ is shown only when its term is below $B$. Otherwise the page says there
is no backup worth a second journey.

**No clear choice.** When $C_{j^{*}} > B$, the page says that no option is a clear choice.

**Break-even for the nearest facility.** Let $k$ be the nearest facility. When $k$ is not
the first choice, the page reports, holding everything else fixed,
"""),
    ("tex", r"p^{\circ} = \min\big\{\, p \in [0, 1] \ :\ k = \arg\min_{j} C_j \ \text{for every}\ p_k \geq p \,\big\}"),
    ("md", """
found by recomputing the model for $p_k = 0, 0.01, \\dots, 1$. In words: how likely you
would have to believe the nearest facility is to help before going there first becomes the
better plan. If you know something the record does not, you can overrule the suggestion.

**Sensitivity.** The page recomputes $j^{*}$ with $B$ halved and doubled and says whether
the first choice changes. It also lists the lowest values of $C_j$ (up to five).

**Limits of the model.** $p_j$ and $B$ are our assumptions. Distances are straight lines.
At most one backup is considered. The model does not know about beds, staff on duty or
roads.
"""),
]


def about_text(text: str) -> str:
    """Put the constants the code uses into the page text."""
    return (text.replace("@R@", str(URGENT_RADIUS_KM)).replace("@B@", str(PENALTY_KM))
            .replace("@P@", str(MANY_PHONES)).replace("@F@", str(FAR_BACKUP_KM)))


def screen_about() -> None:
    top_bar("back")
    space(8)
    prose = st.container()
    with prose:
        marker("prose")
    show_html('<h1 class="sa-h1 small">How this tool decides, and what it cannot tell '
              "you</h1>", prose)
    short, tech = prose.tabs(["In short", "For technical readers"])
    short.markdown(ABOUT_SHORT)
    with tech:
        st.markdown("Each part opens when you click it. Nothing here changes what the tool "
                    "shows: it explains how the lists and the suggestion are produced.")
        for title, body in ABOUT_TECH:
            with st.expander(title):
                if body is not None:
                    st.markdown(about_text(body))
                    continue
                for kind, text in ABOUT_MODEL:
                    if kind == "tex":
                        st.latex(text)
                    else:
                        st.markdown(about_text(text))


# ----------------------------------------------------------------------------
# The home page illustration, stored here so the app needs no separate image file.
# It is a JPEG written as base64 text. Nothing below this block depends on its content.
# ----------------------------------------------------------------------------

# Icons on the two step-2 cards (144 px PNG, transparent, drawn in the plum colour).
CARD_ICON_BASE64 = {
    "high_risk_delivery": (
        "iVBORw0KGgoAAAANSUhEUgAAAJAAAACQCAYAAADnRuK4AAA4xElEQVR42u29e5xdZXkv/v0+77v2noQkXCLh4oWL"
        "CclMQIV4v228AUJmAp5uIZmgrdaoPdp691drO4zVam09Wu2hiNYWYSbg7vlBMgknBS/ZWhUvgBCyZ4CUGkWRcCe3"
        "2Xu97/OcP9ZaMzshWnKfiXk/HzXCnsnaaz3ruX6f7xc4fA6fw+fwOXwOn0l5ePgWZPehD31sVBvctGkTAWBWfZZ1"
        "ocuKD+z872qoKQA7fOt+j42mWq26SqXi9/QX9KFPKpWK70OfHPZAvyfHAJ5dqbh6vR7a78PF8y88VTScQXIugOfA"
        "cByADpICswDicTP8imb/qcD6UtkNX3XHisfbjamBBmuoxcMGdOh6HKnVsge88ISFU2ccw9dScR6AVxg4xzt3hIAw"
        "AGY6dovIsT/BYIiqMOivAd5mhtUI8abBe1ffN/b3oCq/L4b0e2FA1WrVFYbTe8ZFz5IYlxrsD4VurlAQTaEWFWBq"
        "BiPMDCBBGsxIjt8nM4JwhDgnAoCIMWwx4N8FuOLq4ZXfLDxSP/rtUM+T+PuQHPejXxeftvAZ4v2HAX27FzczaoQC"
        "TQIwmNBIwNQymzGCAPOnn1uTwYRAnu/QxtwUmXhxVDPA7OZI+/Tg+pXf2dl4DxvQZPI6qLoijPR2db+V4F978c9O"
        "NZgZmoQJQLHMTmKeICUiIkKC7beGgBmgplAooNYCaWOGxyzmGSGJJN5MoYarTbd/dGDk3x+oVCp+p5zrsAFNBuN5"
        "6/MXHRVTfMmLWxo0Qs1GCbjcJBSgUVj2dDAYUg2bYbwPtPto+LUR22gMoB0B8FlmeC5oJ3vxRwqJqBGq2gJpAJzB"
        "BIZIQhKXuBjj/ar2roGRlavzazrkSv9DzoCKt713/sIXCfy/OkpXGtNWHomEgAI0J1IGiGjxPtC+ZZBvQdNbBodv"
        "/MXveMiszrngxMTLCwV8o8Fe75x/LgwIGtLcbUnmsBiELBOAKT58zfCKvz8U8yIeisazZF73QufcdQSnRotNwHyW"
        "/yIl2eHFIaj+yIAvdkQb+trdKzfv7ME2VTbtcG/y5uEOuczCBQunzhjleTR5tyNfbwBUYwuEy00kgmTJJUkrtP73"
        "wPDQew41I+KhZjyL519wroNbaQZvQEqDM2SVlXdJKWrYCNpfP3f9Wf/Sj34tEl0A6KrVrD97sPa7+khvrlYFANqT"
        "40vndb/JRD7lxc3LPB6ZV/6AmZZ8qdSKrSsGGkPvPpTCGQ+lMv3iuRecXXJ+lcE6zCxmSbJFGCRxPglR/6VVbn60"
        "9rM1D7X93N48SFarVflGraYErGduz/QZDp8g5X1qGpFfA2g0o3b4UtIMzc8PDA994FBJrCe9AeUhQZd0nn8S6X9M"
        "cpaptpAly5GEA0TV7E8Hh1de0e6t9luvqat7MSFfBTDFTFMQLndfseRLpWZo/tng8KovHgolvpvsL8Cs6iw59thj"
        "3YxtU65PxM+Ppk2CDoCSdIA0VbU6ODI0WK1WXaPRwMaNG3VfX0ij0TAArFQqfuinN905f+ac7zvKhRROU7OYdwYk"
        "mkYncu78mXO+ed13b/hFFVXXQGPShjKZ5KFLarVaPHHTjI+XpFRJNYwS5rMSHSAZo6E6ODK0atmCBUn+tu/Ph2X1"
        "ej0sW7AgWT6yqp5a+kYYHqPQGWiWdR9B0ovIVb2zz5uRT/x52AMdjF5PoxaXzF10hnO4OpoqjQIQINWJK6dR337t"
        "yMp/W7ZgQXLlrbemB+rabn3gAV22YEFy1Z3f/cX8Z8xe5+h6ASgB0giFpmUpzQo0+ceHv3Jz7hntsAc6gKerL8fq"
        "OP2MoysBUMDEYNGLK0fVz107MnRVpVLxB9J4inPlrbemyxYsSJYPr75RY/xY4nxiBs2HIGzFNBXI/1zc+aY5tVpN"
        "JyskxE3Wquvyyy/X3vkLz/GSfCLEkBIUkNGJlKLFn90/68mLX7zxxbxx4416sK7z1gcesEql4oduvfm7XTPnvDxx"
        "/jSFpQDEgOidn2oIR6176J7rZ2GWTMZcaFJafVct8z5m/LM811GDAWY0wED5QFuVdTAfis2qzzIAUNgHoukoDJIl"
        "PXQhxgDjRRfPXXRyDbU4Gb2QTNay/ZLTup8nxtfHGAMARzJ68SU1+z8D61esrVbHh6kH89RQi9Vq1V07PLTeTK9M"
        "nE9IKmhiREicn+bElgLA2srawwa0v0+j2iAAOMe3eOdLIGLeO3bRNKrhcwCI2oTzmAxRPp/GsAWwJIeDuGgKEBdV"
        "q1VXr9fjYQPaz32fWq0Wqy+tTjHYhUGjmZkAps45b9DvXzu88kd9wISClvajX/v6+njd3St+DthKL15gpiSpqoGG"
        "08uNbacDsMkWxmSSle4CAOXNrTOFcqqahhzQZQICKldnoaAy4b5Xo9FgjkK7xjLYI5H9d0icL2nkaydjGJtUFzs2"
        "ITd7iXeOICIBg5lPY7olGL4JAGfXz9aJdu3FzI1p+H6w+IBQEsvCmIAERc4EgFmzZtlhA9pPp6hozGyB5TNzI0xE"
        "HIh11929YiMAFlP2CXasD30ysGHNkwRudxQwRzGqRgB2Wh6i9bAB7ceKpg99YkSXmoKAEDShwMDbAVilUpmwva2x"
        "8KS4LcNc0whCYQD4rOqC6oy87cDDBrSfkAP3nvHTIwkcZ6YwUJDh3UHDPZPotd1gbe0py85RybZ4VN6qOGxA+6H/"
        "k93UURxtwHSDKTOwFqMpYNgw0XOI4tpM8Yhae6QyJTmV1poJAA00DhvQfjsd/ggYOmB5BkSKmYHGx/I4N5FjMADA"
        "wTZbtgJUrCwaQdK5jsn2OPxku2AN5oXw2fAUALLFnMi4LzwPDcBlubcrPEE7yUKOZ963IxLL/uZoEg4b0P52mczG"
        "XeP3niYETDhldxcOi6S2DTCfu4R+ezojlYKxY9asWZbjqfW/aWQBNSCC0x0lQ7xa8T0sRqbNnQ32sAHt4+OU26Kw"
        "ZbASAYWZiTiI4tixXlF91w98bWWtFMbSj35DfccHvmzBwqlbt6ZHIinPsDROE+fLAKCRTYpuF+iWUbS2bDq2+WR/"
        "vT/sKlxWKhX/2+hfCnoYEMcJCSUNBoMgMcOTiZQeAYDL0G/9hw0I+3ocYACQ+viEKLaQnAmDkdAMLWqzn2o0kLWV"
        "itTr9diPfi0MJjMUzFORTkfMNbU5RpyyZTtnQUozEO0I0JVglk3OxRRmIVJGEytve+am8qNLunp+RbOfgxyB6l3B"
        "ueHr1q/4ZTvWulqtOtSy9sOO5aTNKTZfjUYHQTRsOmb79ocmG1B90nmgGeUTHt267TcPCTkzQo2W5S2gnVFUOlVU"
        "XRe6LDOaugLA0q7u2QBeq+Drtm6zF0JwUiLiiGzJ2TJnBgWMBjVYVMs63flqvNAwHeQMgscL2UXJHrWS8KpP9Hb2"
        "3E3im0p8c1qH/ujKWm3bzp4ph5+cXsBPCKiIQBl+/qUNa5p9gPy3ofDwVsbeQTl6O7tXeue705htgzqKj6b3JFuf"
        "fP5VG+ujxeeXdL7pJEF6vhFVGF7knZ9mZohZCd00MysMBBkVxy7uRxZn2lbkScAgOQYSVNAIQ+LEiaMgaITB7oFx"
        "tTjWrr7rhh8Wv613du8MKz05LJQTzSwAiIn4ckvDFwYbK98/2dZ9JpUHWltZK6hDDfyZULpz8gxRteAoz007jpoD"
        "YN3S+Yu6zeztQDjbiT9SzaAWQxrDaG4TGV8LSRhImMIy4hZmgC/JOBcoILK1rgyobwRs3KaMBFz2PxaiRVONmuOT"
        "TnNOTouq71/a1XMLBVdNKevXt2x98tWJ+BNTDQGA5N4OAH4wGWdhk8qAipurZt+LGgFSso1lM5AOYp/t7eyZJuAr"
        "kZEfxBDDdss4ooQZlFRppgZ4oSROJLcjg5oiQhVAywwRzAkYaARYIlgWCiSnC8oMU2GwUOCdM5gwRQ2pxhAJOhF5"
        "qYO8dMu29L0kGU1zUhCqwXyq6Wa65IftaMvDIWz/Xa+9bW7P9FFndwvlBFNrZRhjMyeSAEA0bdKguYHBzCwjGmPZ"
        "MTOYEEMEuNFgwySGDbxHYL9UdQ8Z0ydVfepLElqjLetI6KC+w6hHG3EcDKeAmAuzeQA7KTze02X0LxqjkSFrErJY"
        "41EDTMiOrO+sMbvzDI5SDqrfHhxe+fr8++lhD7T/jlW7qqWvNWqbl3QtusbTfbiFVIsXQc2aOSGUGGHMvEjZO+8s"
        "C2O/Dmo/AHWtiP+PqY+m9175wKpte3NBS8644GhT6UqjVkCeA/BFifipMTOmVsZLZSTo1CxllkYJLMOhQAiAV+bf"
        "Lak1aq3DHmh/cRzm3INvm9szven4j468NGpMLf8eWaQyNUIF0uFEEDQ+bMAaqF2fSnNtrXHTo7vqDxUhcscQ0j/e"
        "ECg609Xxfs7aej1yp17P4s4L5jj4RQbrFeELhA4hptHIQBvfgslSLhgpXqH/CqcfGly3+rHJlkRzElVfBsCWzO9e"
        "SONnvbjONIaQE1+OJbcZJw8RLd5OuK9Jml7/9XtX/2pngzm7frbuI5oVVlGVTZVNzDHNVvSAOobTV6rq22F4k3fu"
        "iKAhAgzYyZASlyRRwz2AvfuaxtC39wHpw2ED2gVVnfTO7/kbgXw0C0fWzNnGDICKZAluGuNPQH5h2pTjalfeemVa"
        "/A5Ux1GB+9vY11bWSrsXWXz6wnmi/DOA7/TiGGJo5a0Dlwe44MiyGVShHxlsDH2uvW1x2ICwl4RRneef5Jh8xYl7"
        "QxrTjJ/QzAEIBJLEJS5YvBeGv2k2SlcXnd/85+NBepNZYLiL67m068JXKvTjTty5ago11aK/BEMESJ/R0FzZ2py8"
        "r3Z/bXs71+NhA9oT45nf8xoxLvfijmtpaBZ5hMGCE9dhZttA/L01m58b2LDmyX3E+7PPvVI7CfnS+T2X0Pg3BntO"
        "O7AMRgMQvPMdUcMtIeJ/XHv3yl9PZCPihGYb6+rpdeA/gyiraouEM4MSyNjGLP6HxfhnAyOrbpsMlLpVZExoNdRi"
        "77yFZ1HcLQp1Y15ozIgs9eKnBIvDLVhPrTG0YaJ+N5mI+hX1ej0s6er+YEK5xsyc5fmOgYGkJ+nTGD/xy2c8/pqB"
        "kVW35XoXnOhkTTXU4thmCfFGJ5JkoWsHDwSASaph1FE6S5CbLp73xtNqtWzD9bAH+h3XUsk1LHq7ev7Ki+sPMbTG"
        "KeKQOpEOMzxoam8fGFm5up1IfHJUk5B+wN7adf5xAX4dgJmA6VgbgpSMg7oI0wyeUlbT+9TCaweHb9w40cKZTKCw"
        "lRlP58K/SMT3B43N4sbC0Eqc71DgdkBfOTCycnWhsjNZjCdby64SgKUmH0vEP8PMguXPgFnD/BHnXMmA3KgsiaZN"
        "oZxK+pXVrnOOmWgkDJxYOc8F7ylL+UuphmY+76YBrcS5qVH1m01sv7jWuOnRyUhQWXiOS07vfp5X+ZFBkxzNql6c"
        "DxrXw+kiBrk88cm5rRia2Q49ADBNxJeDpt/1W485d9vGbelEYXmVicD1k5XqCy9OmHwpjWkro2kxIDeeNOpKt/Xx"
        "7lrjpkerqLpJyG5KVLM/uMjPC9nRRjZlAiHMPju4bvV9ftsxF6Yx3FgSX0ZGU2yE+VRDs+RKr06nPnZFDbU4Ufbf"
        "3ESgqVvcdcGLPf3/rzAxQjPDZitx7oigccW0qb+qfm34h60+9MnluFwxCTmsb7zxxtjbtfADzvl3BdNmxh7L4MWX"
        "Wxa+e9rwgg+iArfijhWtk/ycG0od9lJPPyeapQRJmoumacn5BV0zn/vgqp/e/ONKpeL3B2HopPBAfeiTGmp6yend"
        "xzm4/2PAVDOLeZs/zY3n5kdarYuvvPXWMBm6sr/Lwy6et/BlhPtMjDGlwSFzsqKmW0Xt/f3o11n1WdaHPln1wKpt"
        "T0ZcFDX+xIsrGxFhNDOTVEPqxH/hks6el9Tr9XCwKzM5yDw/5pRfFnHPUtPRXGcidSJTQ4w/a42Wq2s2rGlOpkpr"
        "5/tbq9X0bXN7pgvln0l6EBmCETAQPpo2WUruB4BvoKb96Ncqqm7l3Ss3k/oHavqAgElGJMFcysxKQstYXmsHl+VV"
        "Diaz/CVd3R/04hbFGLYD8CAiycQMD1qI/6N2X+2JKqpukhoPK5WK9KGPo86u8uI6o8bm2CCVJlmO549BjK9FLsXZ"
        "zmp2dWPVL8xwMYgUgOTFvqjqaEmSuZaUPtWPfq3m0gu/FwZURW48nT1nesinsl4P3DgcgwLVSwfvXX1fpVLxk1Q6"
        "cqyndW/nrV8oSXJRqnHbWM451jBEgWxcBMDa4ay1Wi1WKhU/MLzie1H1/V5cUgjcGeDTmLaE/JPezkWvOphNRjkY"
        "1UgVVSfE5SIs5z0PQQ4uj6qXXTMydPNk1pIojGdJV/cHvUvem2q6nYTLxhUZrUhRxASNRsM5vfPOPaFWq8W+tmdS"
        "r9dDpVLxy4eH/imN4d8Sl5QMCMzppknQaF+odlVLByuUycFgli91jf5JSfxLg2ozfyuDF9/RimHN8pGhv26bok9G"
        "z+Pr9XpYOq/7o1783weNo0AOrSVURErj1a8xC2PJMZCOPwCeyq5Wz8iyRB3eEzT+2hEJYEqjV9XRRPxZJTQ/cLBC"
        "mRzINn6tVstEUYz9aQwps70oI+ii6WahvRcYYxizyagKnfW0uj/ivP9M0Dja5hKip+tQ1R+a2cNjRgUTzQD9b+tD"
        "nzz1xenXKqq89q6hB6PaB0VcgUYwkBJiCDR85JK5PSdmhOUH1inIgW7jg/6TzrmjDbkcE6DeuUShH7smnzpPtqS5"
        "GC3UarXY29l9mRf/t6mGJs0EZgJD6ihTo+kPW+w4B0A9Ec98kOqihpiIe8G9nXecC8CKqf3OVMHLR1Zem8aw2jtf"
        "AhBzgtHUO3c0nX0UgOX3+dAyoKLqWjp30cuFXBJiaBH0IKIT15FqvOW0xlmX5wn2pDKewuCrqEpvZ8+XE5f0BYvN"
        "fGOWBqZOZJpCf6ZBe2qN2haBDGTrQFnOYqCSBKF/AYC7IlcosNrO3AdUdStJnxcdPo0xiPFtS+a86dQDLZsgB5JZ"
        "XsX+0lFkBw32rB/ysX70a97ut8nUYa7VarHadf7xpfmtVd65ZS1NR5kx5hNg6kWOULP1Kdiz/J5VD/ehT5os3Rg1"
        "rnMiHkAEzAUN0Tn3it55Pefn+YzbmSq4Wq26q0duuCdCv5o4n6sSjXmhafDh/ZkXOnAEVXKg3tBLurpfK8Jz8nVk"
        "ByJ650vRwsDg+pXfmUzia/kbLnmy/IYSkh84yHkhhu3IJDaJrJs+NZr9yFTecN36Fb8sKGFqjVrLYH8nlPG9aeS0"
        "NcQnly1Yluyqqir+GdX/bRrCoxB6gArAhRgDaW9969xFJ2fV3IHxQnKgvI8YPioUyfC/VBhciHFrpL8MACfLRmal"
        "UvF5jqa9nT0fhsiNBE4JGkYN8FlRgFByydRg8dtbtuK8gZHrHygaokWImTb1xGtTjbc5cTmozFyIMU2cf8GWrb9e"
        "1o9+3XlgWlRaAyPXP2DEFQm9gNAsIbc0kWR6dPaOA8k3zQMBYVg6d9HL4ey7aqbZ+rmFkks60ph+ZWB4aNlk8D7t"
        "q0XZloV8wYs7N40hbSNoUILw4sqphq+nTz70rtr9tzwFGF98396unh5PtyLNJMMdM28CAE8I9KyrG6t+sYsZoACw"
        "3nnnHg8prQN4NDKSBpCSKHTjtCk2/8pbV20rNnknrwfKIQzm7F1OvCMQcg4Dn2oYBfmFie59+tAn7ZVhb9ei94rK"
        "Dx3l3BDD9jHjASLBklDKIYa/GGwMvbV2/y3b+wDZuZtehJiBxsqhloXveucTGoLBxMyCo8xUyBVtnqT9Rc+90L8/"
        "AGAgES8gFaCYaerFn7x5G84pGpqTNoRZjlHunXfRCQbrDhrUQE9SvXPezFYPNFY2+vom5qDUcnx2HnbiJZ09L+nt"
        "6l7rRb4Is+lB46jlq+EEQiK+g8BvosYLB4aH/iYvxfnbuH5y/kUrqfuAWmwBBI2RoA8WRxNxb1wyb+H7826021Uu"
        "ZE6+msaQwsxnJQkVBgjlDw8UY7/sKyB88aYWb0sxGCTDhQn9UTC0CDAHyUMMX27TkJhQuOVqteoIWK1WixfPe+Np"
        "vfN7vuxh3xW6V+cUMRGZ8UQA4l0yJZquJvSlg8NDK9qkpuy/k4G6auSGW1XtM4nzJSu0Xo0ujTEVcZ+8pLPnzHq9"
        "Htp7Q/3o1z6Ag+tWrDNgrRPnDaY0k6ghmtlrLp6/6Nn552TS5UAFeBwAlnR11z3dq6JpE2YUcSXVuO6Iqb9ecOWt"
        "t4aJUra3r9wAQO8ZFz3Lor6LsPck4o9sxTQAyLTJsmuOIm6KmY1Ccdk1Iyv+dqdN2qe/fNgFl6D5Q085K79PPpMs"
        "l8QM/xmdvvLau4YebM+HxpcuF16auOTrIYbtuUeM3vmONKTvWD6y6qv7e6bIvf3Zalc1Kdnoh0TkVQCGR5vNT9c2"
        "rHnIAL5l/oXz1PR2gyV5ghgT8eU0pH81MD7zCgd7/NC+hHjx3EUnO4c/JuydXtwzQgzRgLQNvRlIlLx4FzTWTeWD"
        "gyM33FqUzbsbjguD652/8EVi7vuZ7AGYqy+GXP/1P1qby+fU7q+N9uVhMaOHpPXOu+gEY7iLxFEwhBxj3REsrh5s"
        "DC3c35R5spfM8VbC6D90JB2fAnhe4pL3l0qlb1e7zj+egEWL53txZYJpPr9J0hi2BrjrDqKqDqvVqss9jhVS4L3z"
        "F529tKvnaid6eyLyFwY7JsQ4miXHGVoAABPnp8D4UBrjuwcaK18zOHLDrUWutCe5XBHKBtav+olCP+jFJ9nfRSPg"
        "osbt3rlXlmY0l2fAuky0haD19fXJwMj1D5D8lhcvBioBiRqVxpddcnr3cf1jGx4TywMRgC0544KjEWREiJmaWblm"
        "kIx4G4L9gTl+0YksjBpbBuR73+EHg8NDrzgQJebOnNA5I8fYQ75kbs+JzrGbsEsNeIUXhzQGJdAExhqCkYA48aVo"
        "cRuBf03Jz1y3fsUv9+VeWuGNe7t6Lk/Ev7sV0+1oS9K981OCxuvuP/aJpUVOVDCC9Hb2/FHi/D+3Yjqa0+upo5RT"
        "hJ7l61cN7c82yV4RTKVbp2qpYzRYRhInAF2IoeXEnRVcvIW0cjTVvMcRhAIBv9VGvxv2p8HMmjXLarVxTug66lj6"
        "vItmaSu8koIqgNc5kWPNDKaxlRuP5MajJOkoHdFUo2ktin3y2ruG7mzv5fQ/DVLyp3Pq9XqsVqtu9qZNf7rhoRnz"
        "EudfE2LYDjAxmA8xbE+cv/hZDx05vXrq65fU7qs9UX2oWgIQFPF7abTtBIqdMhMKGPEKAENj/NQTKQcqYveSzoWf"
        "KfvyR5ux1QSQjDFNkCUUAKqMjJeEWAzxjOX3rBrpq1T8WozrgOW0vE9HRoDFBy5DHwu2+CIk7sobLO68YI6Ir8Bs"
        "ocFe5uhmkUDQqDQ0jRSYCbMyOBqs5J33UaOSGHKwv7tq/dD39zdxQ5GvXDrvwpkq+i1HeX6GJ8oRm0RIJJkSNN6W"
        "amvxdSP/955lC5YlJ9x6Qry367YfeLqXFMWKE1+KGr4zMDz02v25kMC9fdNRWSsbHjqyVnLJhc2QjpL5MpzlXVWa"
        "5AYkMGwD+Uf07rvX3Hn9pv/ud7er1rQZ2O98cMsWLJy6dbucbKYLhPJSg73YjJ3euSNoQLBgBrRgsIx0kwQs5jIK"
        "HU48QgxbQFxP8opr1q/4QTtkY3/3rIqHffH8Rc9ODDeTnBs1bm+LFtE716FqD6vpHw8OD63Iqt2e/52I+5O8zeBI"
        "JgY8nKI8t9aoPdqHil9byTzdvjR+7ouff+/s80qPlUpf9eKXZvS1Nv572/G/NBE6UdVHQbvDgJ8aeAdpP49BfnUE"
        "7JGv3b1yy3/3Bd87+7zyQ640E2bPEGfPpHEuHGab8TSanQryWY5SJomMq1ADwWAwZCEqh4SaKcCyE5dRjVu8G+By"
        "qiy/euSGew6k4ezSu59xwakS3RqhzAkatgMFhAMRRCknK/58c3rp46Unt13gpVRLYxjNoTIg6aPFVy1vrPqPXcBr"
        "9okX5T4shWPvvO73ibi/U+iYbklGx0K22xNJ7+gg+ZZKVIWZbjHwUcAeI/CoAZuRlc/GjG21ZIajCJsG8ggzO5bg"
        "NBHxjgKDoSARzwm8AzPAiBTYnHzcQNBKjp4kkGp4nOC3SdTgtg9dc+fNW4uHOMZ2fxBXoS+ev+jZ3rDCi5xZ5ESF"
        "xhgAJC4ppTHcabR/J/inWQ8pSxkyllg8AODXNN4D2LdLaWvoXzaseag9ZB5UAypc7pLOnldT+Oc0vN7MdiWkxB1U"
        "lcw0m6taJpaVv11kRv9cJDrjBExZ0WZ5iqRmOauXaV6+ZuPMQoMrYzGzvOnnQJa8eACGVOOThP0YxlVBuSKT4x6v"
        "hn5bLnWwjKjadc4xZXZc4+nfmMbQzKLv2LOLjq4DNKhqK5dlyOD2AIUiefGSNbEsPADl/2oOlz5fEDXszXflvkAa"
        "9nZ2f0zEfVJIBg0x74VZQf5eGFD+5RRAzBtlzEzGxpF5bdZnyMikcoMQg0kO48sVB4AMzlA44zGqFC8ULxQUybIZ"
        "7ietbmY3R3XfazeaMZzOBCS2LB5wH/pkQ9ftn3aUj0RTM7NRZPCRtqIjw1kXL2sbe62amZFUAOXEeYQYvmeutWRg"
        "3Zr798aIuNcw1c6eTyQ++ctWTEPmVeDaPIxkusq5QjoAoYgXBzNAc4ETMzUDIwpWBeZ5k7XpQuak4W0qf5lKGMUJ"
        "iQwSmoXEHB5xH4GfmfJHBrkltJJG7b7aEzvvbk0Ub/M0R0O2eH7PJWL4By9uVoihlXOZu0yvI79vTzWg1GCSjWGo"
        "BgslSTqCxuFmGt5Qy1hsZU9Izrk3xrO4a+E7S5JcETQ2zUyyfSUzkkkiHmkMeUmfiU14513U+F2o3WDCl8FsLogT"
        "AUx3lA6hZEYA7FDNj0kjodhHyCUGDNsMeJTEL2H2Xwb+pxkaDvGuh0O8d82GNc2dQ8KmyiZOBqP5Xblm9bTuU8qe"
        "nwC4lCRUYyt/QYs5HcZCGQyJJIwWEVVz3BEMhpZ3fkrU+JNSxOueffeZW/eE9ph7GJd16fwLngfzP8mS1wwADJiK"
        "uJKaPmGGfyHtQoAnGSzSoCVfKrVi65MDjaG/LH7fpfMunBmYHuONJ0a640GbQcNRRkyloay0RDKcwhYBHwPtcVM8"
        "YrBHnPAha6WbCnLNXV3rpsom/jYBuMl42rvKvZ0LX0dKn9C9CrSsr5XlioWslFOzhwT8igFv8uLmpRpSWhb+jRbK"
        "rtTRDK2vDg4PvWNPOtbcY8mlru7Vjv78YHGUsASG6JwradQfg9b73MZZ923ovP0XIJ4JIMBME5+U0hg+eP+xT3xx"
        "ypQpbs2aHT3E3r6dOzUU7VAwmN8W0hqojrG+9s7ruYjCD4L2csuEZMissPAK/GqwceZzlj7vB1MsdPy9F/+ugjow"
        "o6jOPieQl3+9ccOPd5dCj3uWNC96FQV1U03zRFgJJgZsjKIvu/auoQeXntZ9ijncDmJ6Xj7FxCWlVoxvGRxecXX7"
        "9mnRNBwjoMTTUO2p7dBcxKFqLE+ThB2ZR+r5sBP5bNCYkhCADoZHW9Qza41Vv8g/82Xv3LIx/klY6p2fkmq4YbAx"
        "dNHuJtR7OAuzdzk6hmx/JVcZICzGd13bWPVgH/pkJLlDnGqSbe1kSbTBQLMtbdJNtoMScn33UYNrKxV3dr2ul+Ug"
        "tome37Sz5vflo5g9rf4K46l2VUuZSIveZpAxt2AwJVBKUvOFl279EO+z6aOvdCKdqpoC9EFDJHje0q7u2f2N/g27"
        "Y0S7A+dgrVaLi09b+AwQ5wSNlhdN0YkrR9OVAyOrblq2YFnSj34tASVynAGtrZnY2lflLQGr1+uhPy/3sz/360Qi"
        "odwFqVYsICQFXLYQ/t7jX9zIqIJj1vvaObI4K+dzyfVwtftr2wn0E8JxmhmGRHyHAW8Edm+jw+8uMQIcXukoz4ga"
        "W3kpzUwH0v0TAD62/THuFCPzqiBrAqpT2Rd4ZaJfq13nH58weTdNX575QNweKFf0r+//z4nGaFZcz+J5CysU9xYC"
        "p5DYbIYbW43SV3NvslcQF8Ja7cz3eSeeovkyZ2ZobLHjxkRHfykiz1a1lCxatHwdgC+dXT9b608zHDzth1kkqCRf"
        "KMzGSZb1dcpR46/QHL0FgGF+9jakGgVjG5oY6ysTtH3heZbOv+D5HUxuKdH9ldC93lFel4j/UAL78dKu7vMmkica"
        "13rt+bB3fq0XeZsQrxGwpyTuilLX6JplC6pH9mX3ao89kZjjU140g6lkmhxd6LJqtSq1Rm0Lwe9l4ntQmDHmH3nr"
        "SZWO/MXjPjWgQnEYJnPHRWjNRAQgRgY2rHnS2lZ06CwauVNLh2AsZjl79pJdhn47b/Z5ZVP5Z6GclMZ0S7Q4Gi02"
        "WzFsI3iMAl9bcsYFR+e5FQ92zpN5ngtfJsLPRo1piHFbNG1G1WYzhm1lV379lm3NT/bnKzt7uj4F4RQZdydF2hBd"
        "i82dHAEB/Czvr5nlnV8zHMdpRx3bhjjddwaU91FA2jMzmr5MojZr/fHnAPDmti/vWmzCMjKkwp0SBAVHtXu03XwY"
        "QsCOKZdeLuIWpBqaADtgcMi0/0rRdDSR5ASJ7nwAdrDpcIvKUhj/KGuUIiBre2SjGVjSDM1I4uJL5104M8+J9sjo"
        "Te1oZlOvzCaynnTT+bh9h2Ilm2k/aGMt/Uyoj8DUEOXI/YGJJnIsLgxH5OUTx+tne7QwiuIiQwnbSGzPfpZjmZ0Z"
        "Zuztw4BhtmTLdJng7k5+GzBTtc6JEL4K3LeRJ2fTXUrhHbL/jA2Xj4liz96dt/+p6UU8itxhkAwatoWWbttZuNgE"
        "2wrt+nxybyB88LFjv4HqG2jQYL4wKT61whrPzmV0C4Ans/nU+HyLwEzspWozTZ7Ir32nmG+WaZKCJnwUE0LeYAwU"
        "91iunzpWVBT2bpkj3+bT8BB2kNrcXRfEEwr1scLjg3yytWH61qd4BOouXjyYhGj7w4Cs6DuQ2Ma2tl3+wh+5s+b5"
        "NXfevM3MHiV37PEp8Jw91Uf/RsEdZPxeqvFJATyJMH5TMqaKqDEy8uaDuPnxFO8A01WSVR9KjPkeIxFKkogZf/j1"
        "e1f/Kku4d2+oOfZigSePjVBJowjUbFN7hVdcj0Y5JndbeTAxwmw0Itm8XzxQUdGY4YFs7o2iDwQjT84e8Dd0LNRl"
        "VvPrPO7neu8Kwp6T74fvyeTXqqi6bJVF/9yLd6RMafOC5cSXSlHt04N3r1g3EUr5er0e+9AnM9PwjWZIb+rwpakg"
        "fC7zKhnGWR+n6Z/v6d9R3EuFzcnzU0HOC0ziv4o2zI7Oip0FvNwMRtKBfLSjbJt2ypf2jQG1NZfuzVFfBlBi1lB8"
        "waXzLpyZl+gc/6xt5HiUETUFiFOqXZVpe9o8K0BQA+tXXZ7G9FIqGiBiTsR9dxrDu5aPDP3lBOoDWT/67Usb1jSn"
        "TbWLWrH1BQAPZbBUbkljXKNovSYXzduTFSECsEvnXTgTwOxomkFfxmAv2NDuCYvxkcBebKaAmRB5amb4r6vuWPH4"
        "7vSjZPfDLG+zHX93y4s7VkVf8RRLp9zd9pNihgDDiUk86pQ9SRaxw254nwwMD10zZepxL4iMZwE46+hW8/kDjRVf"
        "3sMHsZ+5GsArb1217ZrGyvdr0Pnm5YUCfd7A8Io3DjZu/Fmb197tyjRnQJlL8liYtQqAnpoChkYR5ooosrjzTXNA"
        "nBUt5jyV1AxTZbcCu8fq4Z9+NVHXOoAkjWtTb08QmA4w5pBUg+EjlUrlxnbCbBrvDBozXHL2rkRHV05N5wNYt7ay"
        "VlDfM0xuIQmQKzMP7wrugAlH+JHtwi+/p/YwgIeLZt9le7GcuKmyiagDavElJSkx1aC5VJYPqlsRdB1yRo/8fgcw"
        "fY+XZEqwtAWYY6bZgWhy0w49v33pgfoB7UOffP3e1b8C8W3vxtmxomorcf4VJ26a8Y5arRaPv32zzzqjbKjpw0L6"
        "tkQXFLwE+0hCMl8Bkvztmuiyl7bjNUOYz8T2uDIda/Dy5TliUzJclghh96b3Tt0IAA8seMDV6/VwSWfPmQ6yLMQQ"
        "YBQYIkWSqPrLrWrfa+/57Xs4Rw4fuHTewgqc+07M1mWYbQiQQm5X1XMHRoZuqVQqftasWZbcNfpN7/zZMYaWEXSU"
        "JKrdOWfWEwv6s81U/j5CMbDv1rJs4YKFU4/czvWknGyqLZCWiC+3NP1aOr9j2fG3b/Zf2rCmufR5F81CiN8EeLqZ"
        "pjkmKGOL09A30Fj5id314LInCezVI6vqUXUocT4xIhTgJAOmU/h/l3QtWlKv10OtVouEfVOyfoQSEFWNBDrvfmRG"
        "197kQYfPWL7JI7fYi0k5KYdnsKiOxWyoVqvFL21Y01w6d9HLLcS1JM9Q01yfxFQopVYMv9iS8Iu5B9cDwc5BKj8S"
        "VbdJobIDOIVGA47y5EBvZ8+3euct+mMjNVq08QYkQ+J8ItHOOZBkkL9l+1X2dlbWhz4p2MgOVlg0yoWejiRjDiF3"
        "IcY0Ck9Y0tX9h0u6upeb6HeE7Iy5rFZeBZoXERIfXHHHisfzhNwOIKi+pzcRuSbG2MqVHzxBhWXYaJKIGlGsDufT"
        "4eCcKwUNt8xpLHjFZeg3HuAQlu8E2W9D9u3ulP23/f8DEb6qXZVpJZtxBymnqFlgPhcBACeSSL7MEmIMxdpTzusY"
        "yq40tRXSzw6MrPzonhYfsoeNq4x6v7FyIGr8UyeuBDLBGA8QGC2OhhhGYUh3xAZRomoQuBff0/XTF7KNHexA3fjc"
        "ePiHnW+a89bnLzqqTQmZu8Fmxn7065I5F5zaO3/R2dVTX3/k7sAg9lH4guf0c524U82sxTH5cEjGE2Rp0NgMGptt"
        "az4RAMq+NLUZW5cPjKz86N4oBMhedD9jFVU30Bj6UsviEiEfK4kv5+91zDvDLmta5+3yHAALIDpxIiaXALAxOAIO"
        "jKbF0tO6T1nategHgeGOkNpdSzp73pU/fKtUKn4XxkRrI6Yq+A+XzO/5OBP/M5h9u9xxxG2987pf2taJx4Hh3+bb"
        "s8bujmvAIDMKMyuesaUAkDjfISKuFVt/NTg89D8L6dE9LWS4L4kAnPqPm9kl3vkpY7tb+fwjv7qQIxiVhBjwiAqe"
        "d+1dQ5v6fgej6b4Gofd2di8v+/IlOSVNWShQsy+Wo338a3ev3LyL+7PDzV06f9HLCfw1wddG1RZgLe/8tFTjbSnK"
        "L6s1aun+BPoXnNWXnN59hjf+xMxcG4mFQZCM7aqCKBYvo6oBdpOpXTYwMnRL+8IiDgbB1BhFG6pucF3tPgBvW3z6"
        "ws9aDOcDONtgp9F4VPaioAzYNBqVGRatlTg/K43xHQA+ubZScajX96cBsQhV9+K256cZaSYMaKoqvHN/2kI4d0lX"
        "91dMbHXQKfdlQPXsgW2Y/aMTLSm/grQ3m1qPiPNBw2i+zZ/khOOndFg6HcAj+zOUNaoNogaViPd650uphZxRzQDC"
        "wfBIsfil1C2m+C+AtxBYcc3wyh+NA932vmfGfflWNNBgezLaV6n4ex+dNt2liaQanykOdQLTx3beSW+GB5IST7/q"
        "jhVP7O/1nPGdtp7rE/EXtmK6LSPFypB7JMpOHILGbWa2keAjoCkMM4x4tqebCRIhhhSk0uDyhLVVcskRrZgODQ4P"
        "LdqfIsFj3qezu8tTfmLQUrFKLiJJNLs9RbgAANDSePSRfmvOWr8D99K+uj6/r75YcUF96JMxPsJ6fwDwWP6RR3o7"
        "e77RtpOUAEhLzj8zbbX+J4BPHSDWVqrFjwTFC724ZwXVbQASAM7MWmkMCqAklE7JYU5Kg6lqhDahliH4LBOMgcFK"
        "LjkiaLhNHd6R3wvsd+8DXObETQlRW0bzBFQojKafqzVu/M2uNFyLlad9RcuHA1AxEAAqqLg66nHJ3EWni7efqlqx"
        "9G8knAGPQ90ZAyPX/2Z/S3wXXuiSzu75jjKQiHt+GkMALEU2csl2Y3LWD+a7DeNEBTkYk4CQZYKIGgd8Sd6zu5Ps"
        "PSa0mL/o5TB8z8xiziiReR+N61JOeREaiF2oWf/4ddhkVesxAFZHPVSrVRm8e8U6M1yXuILKFqJmqRc305h+GgdA"
        "66qY5F87PLR+c7RXpRY/DyJ4l0whM8A/wUiDZjwOZnnIDSDUiMQ7V3Z0ZTO7LUZ908Dw0NKr7ljx+J5O1HeLlxtV"
        "Z6p/Lxn+wsbWqiAk5a9rjVoL1Wx2ib1MkCeCB3pK7F7cecFsofsZDMn4OpOpc74UY3ruwMiqmw7ERL296bdk7qIz"
        "KLYMxEKSJ3s6tMNOixNNYWoPgvi+qQ386rgnV9br9dCu5IP9KDNVr9dD77zu9yU++XwrpqNjqo905ajx+7OHz3w1"
        "+oD+/gMHZeHBYJZY0tXzqZL4j7XfBAGTaDoybaq96IRbF4weIHIEVlEdU9NZ+rw3HKGx4wViONOAOWY8ErSUhs0g"
        "76fauqh2+/J7Vj18IOEjY0uJnRfMcXQ/BTDVCu4kwgTiI7UyuH7o+wcazsKDMH9io6sxNcHoTwUyx0yzYSxzGQRN"
        "vzbYGHr7gZRBaGMoi7v5eT0gRl6tyubbN/uZpdJ3hPIyNW3C4MaJEeI/DTZW/snBwELxYPHbLO5cdL4XrlbTpuUY"
        "XoAxca4jjelbBodXXX0QtDRYRVWK9aECGFfAQQ8Gz1BxDxZ3dv+vDpe8vxlDRqeTDUM9YL8Rdc87deSGx/rH9hYO"
        "YQPaIZR1dl+ROP/ONIbtOd9fQZs4Gkxfdu3w0Po9HXQeCmdMkadr4RJPPxA1trJGIXO1IClr1DcNjKy8/mDdJx5E"
        "KAUbpzamlzqatxCcq7AWzByI4CgdarauNVp+Ve2+2hMTjSjhgHqeeQsrzrk1puay9TqIAWlJSh0tbf3DYGPofQdT"
        "9ehgYXGsgQZr99WeUMS3AogZtSIjDE5Vt3txZ5TKzesqlYo/yMZ+8DzP3EVnOHH/BrPS2C45kCb0HS1Nb0m2Pvn/"
        "VatVV2xa/D4ZEGqoxUql4pc3Vv84mn3Iu6Q0zsrKJI1h1Dt37rM3Hfn1fvRbjlyU3xfjWdrVPZsOK0A8w4AUhLNs"
        "dakUTR+MkYuv2lgfzafyBw0SfFCJBzZu3KiVSsWv+ulNP5z/jNnPKbnkxdE0ZcZ3TDNtJS45c/7M2ad86eErry/4"
        "AeuHKIZ6rNczt2cuRdaQOFVVRwn6nCmZIlRjuHD58NDPqtWqu7xxuR5sUPbBPuzr62Oj1vBlNG904l63Y1KN6J2f"
        "klq8Kl1ffnsx/T/UEuvCeN4874LTS+KGSDk5Zko9PoeqZlxMFnoHG6sGJ4Da44TKKzJ45qnVI8vl5s0i8qKQK9Tk"
        "lVnwzk2NUVeW1JZ+7e6VmyfKDdwX3z2fFYYl87oXisi/kHiGqjUL4LsR6sV3pKH13uUjq/9xIn33iZJTWB/6pHZf"
        "7YmguDCaDjtxUwqILGA+jWG7c9LTdPz2xfPPe24uh+0nc3JdyG7WUQ9LO7vfLSIrDDZTzZqAectDdSK+I8Tw0Ylm"
        "PBOushkXsTv/JKG/iZTTouo2ZI0zEAgiborBfmNm7xxorFx5EMDs+8brVCquXq+HaldlWglH/Z0j3xVVU8vwR5Lj"
        "pcSL86mGDw02hj43Eb0uJyr38cXzz3m2t44hn8EtRgkWiLtIsENIqNrnWix/rNaotdp4p+3phMuJwO3cO3/hi6Dy"
        "lcT557c0jMKsKGoiiZJQEE3fPdgYunKihmw30S6ogYZVq1X3jbVDj59+9Jx/A/mikk/mRNPAIuwyZ753yavEwmu7"
        "jj7tp6tvu+mBIhnduHGjTUSK3zrqaKChyxYsnNp5TOfHYfyqUJ4VNBeJy05w4qYQfFxhiwcbQ4MTOd/jxKXzz8LS"
        "ebPPK89MSld45/8wjWmakUiZz2Ew0YuUo9k2wP7RXPzM4LrVj+1vbdPdna21C9ct7lx0oaN9wok7I40xkIhm5piJ"
        "5WrifEdUvR0a3nLNyOq7JnqxwImtCdGG2elc9CEhPk3Cq2prTBsLiID5xCUujeE/Qf6d3/r4VVdtrI8WhtRVO7DK"
        "g7tSpV48b+HLKHKZI88xAzTbEHVZd9kCgFLivIQYBpCmfzKwYc2TFVR8HRO70uSkkDnKMTu9nYteB8EVCd3sNKat"
        "nExybJJPosOJQ9R4J8gveY9/y2GmO6sR7vPu7Q5Y8NxYly1Ylmzf9uAblPZ2M+v24pJg2ZIfjc4yijv14srR4iNq"
        "+PPBxsqvTKbCgJOt0ZbJP075DIF3wIBoOpqpL0OKMCCUDkdBsLgRQA2qy3MGsB0QAZs2ZVJQT1cRum3JEG9ug33s"
        "nLxfPHfRyc7rm2Fc4sjn57jplmVQC5evFkehTHEiiKqrAX3fNY2hDQcC3fh7aUA7VzBLOhe+kXSf8uLODBraJSCl"
        "WBsiWcrWdEIE+GNT1ATx5l+ObBn5baGhWE/aeVs2D4O/9cEuPa37FPNyjsHOo6HinTtaVRGhzUJiPPeYSqLkxbuo"
        "caPB+gYaQ1dNcHKsQ8OAdg5p580+r3xMqfweMXzYOTkuaDSzrINLQnLt+gjAi4gXEvme+AaY3WrGHxv0LlJ/nmw7"
        "9sGrNl41+nQ84bMe7DiWLD/HaPNgfCGABUY7PRE/3QwIGpRAvuwHglAaI2hlL94FjY+QuJxR/uHqkRse6QME6MNk"
        "hKzwUNDKumRuz4lO8B4Sf+zEHxs0ANkDRM4IDzIXngW9SCY7DgBBIwy62YwPEvYoiIfN+IQZtolYmm96dhgwHYaj"
        "AMwieAIFRztmq/9qiqiqINJcaVryMKWZUD078oXFhwEO+ojPX5WL/k5Gr3NIGNDOOqIA0HvGRc9C1HcCtsTTnWrI"
        "vAFyiSkWo5s2DQ8zCAkvICk5gTHYthTfJvdhmeaQmgGG1GAx2xqj0MaEbjVnbSt555lr2Y/AcA28u2pg3fX3T6A2"
        "w++9Ae3SkHrm9kyf4dBjwBIYXu2dm2ZmCBYNQCtnjJdx2XGg4NRhFvbGZeozFWADbGxumMvrWU4uasXYAUDZS+HZ"
        "wuMwfstgV6ebO26q3V/bPt5WqFn/AcYuHzYgPD3iqDe3GRIALO3qnm2Uc2D2RsBe5uhnCgnN5MahGXVIAMxyCW0r"
        "7KbNEY27o8ygxGhOQMm06YlcgO8+AD8h5d9T4lvXrV/xy/bcaW29HnmIYZkOVZgoq9Wq7NxAfGvX+ce3mLyIai8C"
        "7SyCpxlwnJAzXC45Ph62uMsbpDBEDQbD4wb8huQwDLeo6X+4UvPOa+68eetBWv85bED7q5u9c4NvLBHvqpZKLsxi"
        "S59jTk8y8HjCjjFgOg1TC05HhW0VcLPCHhXlr6LHL5Hi1zOmHf9QzlO9C236uh4qYerwaQtx1WrVtbGQ7ZNqsFKp"
        "+INMtHnYAx289SKwgSpR3VEEr11NaOd/3lXrspwcFIc5rg+fw+fwOXwOn8NnUp7/B0FNHGkAqSKwAAAAAElFTkSu"
        "QmCC"
    ),
    "nicu": (
        "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"
        "AAAAAElFTkSuQmCC"
    ),
}

HERO_JPEG_BASE64 = (
    "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"
)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------

st.set_page_config(page_title=f"{BRAND}: find maternal and newborn care", layout="wide")
# The span keeps this style block out of the page layout (see the .mk rule in the CSS).
# Streamlit draws a popup outside .stApp. Make the text rules reach it as well, and give
# the popup the page's fonts.
_POPUP = 'div[role="dialog"]'
CSS = CSS.replace(".stApp .sa", f":is(.stApp, {_POPUP}) .sa").replace(
    '.stApp [data-testid="stMarkdownContainer"]:has(> .sa)',
    f':is(.stApp, {_POPUP}) [data-testid="stMarkdownContainer"]:has(> .sa)')
CSS = CSS.replace("</style>", f"""{_POPUP} {{ font-family: 'Atkinson Hyperlegible', sans-serif; }}
{_POPUP} [data-testid="stMarkdownContainer"] {{ font-family: 'Atkinson Hyperlegible', sans-serif; }}
{_POPUP} .sa ul {{ margin: 6px 0 12px 0; padding-left: 22px; font-size: 16px; line-height: 1.5;
  color: {INK}; }}
{_POPUP} .sa li {{ margin-bottom: 4px; overflow-wrap: anywhere; }}
{_POPUP} .sa a {{ color: {PLUM}; }}
</style>""", 1)
st.markdown('<span class="mk"></span>' + CSS, unsafe_allow_html=True)

if not WAREHOUSE_ID:
    st.error(
        "The app has no SQL warehouse configured. Add a SQL warehouse resource with the key "
        "'sql-warehouse' in the app settings and redeploy."
    )
    st.stop()

# Streamlit forgets an input's value when a screen without that input is shown (the method
# page). Writing the values back keeps the chosen service and place.
for _key in ["service_label", "place_query"] + [k for k in st.session_state
                                                if str(k).startswith(("ask_nicu_", "ask_high_risk_delivery_"))]:
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]
if st.session_state.get("service_label") not in SERVICES:
    st.session_state.pop("service_label", None)

mode = st.session_state.get("mode")
if mode in MODES:
    screen_results(mode)
elif mode == "about":
    screen_about()
elif mode == "service":
    screen_service()
elif mode == "needs":
    screen_needs()
elif mode == "ask":
    screen_ask()
else:
    screen_home()

# Questions about the data are for planning, so the button stays off the urgent path.
_urgent_path = mode == "urgent" or (mode in ("service", "needs")
                                    and st.session_state.get("role") == "urgent")
if GENIE_SPACE_ID and mode != "ask" and not _urgent_path:
    floating_ask()
