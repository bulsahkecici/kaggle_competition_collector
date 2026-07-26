"""Streamlit web interface for kaggle_competition_collector.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
