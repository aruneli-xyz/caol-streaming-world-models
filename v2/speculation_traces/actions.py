"""Authoritative Solaris VPT action semantics, kept dependency-free.

The ordering and conversion below are copied from
external/solaris/src/data/minecraft.py.  This module deliberately does not
import safeswm.gamma.actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CAMERA_SCALER = 360.0 / 2400.0

ACTION_KEYS = (
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
    "cameraX",
    "cameraY",
)
KEYBOARD_KEYS = ACTION_KEYS[:23]

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}


@dataclass
class VPTActionState:
    """Episode-local state required by the Solaris converter."""

    first: bool = True
    attack_is_stuck: bool = False
    last_hotbar: int = 0


def convert_vpt_record(
    record: dict[str, Any], state: VPTActionState
) -> tuple[list[int], list[float]]:
    """Convert one VPT JSON object to Solaris 23-bit keyboard + yaw/pitch.

    Camera output is ``[cameraX, cameraY] == [yaw, pitch]`` in degrees.
    Hotbar changes are edge-triggered and the episode-start stuck-attack
    correction exactly follows Solaris.
    """

    mouse = record["mouse"]
    new_buttons = mouse["newButtons"]
    if state.first:
        state.attack_is_stuck = new_buttons == [0]
        state.first = False
    elif state.attack_is_stuck and 0 in new_buttons:
        state.attack_is_stuck = False

    active: set[str] = set()
    for key in record["keyboard"]["keys"]:
        mapped = KEYBOARD_BUTTON_MAPPING.get(key)
        if mapped is not None:
            active.add(mapped)

    buttons = mouse["buttons"]
    if 0 in buttons and not state.attack_is_stuck:
        active.add("attack")
    if 1 in buttons:
        active.add("use")
    if 2 in buttons:
        active.add("pickItem")

    current_hotbar = int(record["hotbar"])
    if not 0 <= current_hotbar < 9:
        raise ValueError(f"hotbar must be in [0, 8], got {current_hotbar}")
    if current_hotbar != state.last_hotbar:
        active.add(f"hotbar.{current_hotbar + 1}")
    state.last_hotbar = current_hotbar

    keyboard = [int(key in active) for key in KEYBOARD_KEYS]
    camera = [
        float(mouse["dx"]) * CAMERA_SCALER,
        float(mouse["dy"]) * CAMERA_SCALER,
    ]
    return keyboard, camera


def pack_keyboard(bits: list[int]) -> int:
    """Pack the exact Solaris keyboard state into a lossless 23-bit token."""

    if len(bits) != len(KEYBOARD_KEYS) or any(bit not in (0, 1) for bit in bits):
        raise ValueError("keyboard must contain exactly 23 binary values")
    return sum(bit << index for index, bit in enumerate(bits))


def unpack_keyboard(token: int) -> list[int]:
    if not 0 <= token < (1 << len(KEYBOARD_KEYS)):
        raise ValueError("keyboard token is outside the 23-bit range")
    return [(token >> index) & 1 for index in range(len(KEYBOARD_KEYS))]

