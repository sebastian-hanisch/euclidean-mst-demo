"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import emst_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _float_choice(options):
    def cast(value):
        value = float(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _choice_from(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _bool_from(value):
    text = str(value).lower()
    if text not in ("true", "false"):
        raise ValueError(value)
    return text == "true"


SETTING_SPECS = {
    "kind_select": SettingSpec("kind", _choice_from(C.KINDS), "uniform"),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "clusters_slider": SettingSpec("clusters", int, C.DEFAULT_CLUSTERS, C.CLUSTERS_MIN, C.CLUSTERS_MAX),
    "terrain_select": SettingSpec("terrain", _float_choice(C.TERRAIN_OPTIONS), C.DEFAULT_TERRAIN),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "layer_select": SettingSpec("layer", _choice_from(C.LAYERS), C.DEFAULT_LAYER),
    "cut_slider": SettingSpec("cut", int, C.DEFAULT_CUT, 1, C.CUT_MAX),
}
PRESET_KEYS = {"kind": "kind_select", "n": "n_slider", "clusters": "clusters_slider", "terrain": "terrain_select", "seed": "seed_input", "layer": "layer_select", "cut": "cut_slider"}
STEPS = {}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


WIDGET_KEYS = {"layer_select": "layer_widget", "cut_slider": "cut_widget", "clusters_slider": "clusters_widget", "seed_input": "seed_widget"}


def store_from_widget(state_key):
    """Callback: übernimmt den Wert eines nur zeitweise sichtbaren Reglers in den dauerhaft gespeicherten Wert."""
    st.session_state[state_key] = st.session_state[WIDGET_KEYS[state_key]]


def push_to_widget(state_key):
    """Ist der Regler gerade sichtbar, muss ein geänderter gespeicherter Wert (Preset, Würfel) auch ihn selbst ändern."""
    widget_key = WIDGET_KEYS[state_key]
    if widget_key in st.session_state:
        st.session_state[widget_key] = st.session_state[state_key]


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]
    for state_key in WIDGET_KEYS:
        push_to_widget(state_key)


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
    push_to_widget("seed_input")
