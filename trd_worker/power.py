"""
Cross-platform 'are we running on battery?' detection.

Returns True if running on battery, False if on AC/wall power, None if
indeterminate (treat None as 'on AC' to err on the side of accepting jobs).

Implementations:
  - macOS:    `pmset -g batt` parses 'AC Power' vs 'Battery Power'
  - Linux:    /sys/class/power_supply/AC*/online (1=AC, 0=battery)
  - Windows:  WMI Win32_Battery.BatteryStatus (1=discharging, 2=AC)

No third-party deps — uses only stdlib + subprocess so install stays slim.
"""

from __future__ import annotations
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def is_on_battery() -> Optional[bool]:
    """
    Returns:
        True  → on battery
        False → on AC / wall power
        None  → couldn't determine (e.g. desktop with no battery)
    """
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            return _is_on_battery_mac()
        if sys_name == "Linux":
            return _is_on_battery_linux()
        if sys_name == "Windows":
            return _is_on_battery_windows()
    except Exception:
        # Any unexpected failure → treat as indeterminate, NOT as battery.
        # Refusing jobs because we couldn't read power state would be worse
        # than the (rare) case of running on battery when we shouldn't.
        return None
    return None


def _is_on_battery_mac() -> Optional[bool]:
    out = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if out.returncode != 0:
        return None
    # First line is one of:
    #   "Now drawing from 'AC Power'"
    #   "Now drawing from 'Battery Power'"
    head = (out.stdout or "").splitlines()[:1]
    if not head:
        return None
    line = head[0].lower()
    if "battery power" in line:
        return True
    if "ac power" in line:
        return False
    return None


def _is_on_battery_linux() -> Optional[bool]:
    # Look for any AC* power supply with online=1.
    # Some systems name it ACAD, AC, ADP1, etc. — glob covers all.
    base = Path("/sys/class/power_supply")
    if not base.exists():
        return None
    found_ac = False
    for child in base.iterdir():
        name = child.name.upper()
        if name.startswith("AC") or name.startswith("ADP"):
            online_file = child / "online"
            if online_file.exists():
                try:
                    val = online_file.read_text().strip()
                    if val == "1":
                        return False  # plugged in
                    found_ac = True
                except OSError:
                    continue
    if found_ac:
        return True  # Found an AC adapter, but it reported offline → on battery
    # No AC adapter at all → desktop / VM / no power info available
    return None


def _is_on_battery_windows() -> Optional[bool]:
    # Use wmic for stdlib-only access. Win32_Battery.BatteryStatus:
    #   1 = Discharging (on battery)
    #   2 = On AC
    #   3,4,5 = Charging variants (on AC, charging)
    out = subprocess.run(
        ["wmic", "Path", "Win32_Battery", "Get", "BatteryStatus"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    if out.returncode != 0:
        return None
    lines = [l.strip() for l in (out.stdout or "").splitlines() if l.strip()]
    # Header line is "BatteryStatus", then one number per battery
    statuses = []
    for l in lines[1:]:
        try:
            statuses.append(int(l))
        except ValueError:
            continue
    if not statuses:
        return None  # No battery present (desktop)
    # If ANY battery is on AC (2,3,4,5), consider us on AC overall.
    # If all are 1 (discharging), we're on battery.
    if all(s == 1 for s in statuses):
        return True
    return False
