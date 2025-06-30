# modern_style.py

import tkinter as tk
from tkinter import ttk

# Farbpalette für modernes Look & Feel
PALETTE = {
    "bg":      "#111315",   # Main background
    "fg":      "#f4f4f4",   # Main foreground
    "accent":  "#028ac4",   # Accent / Buttons
    "border":  "#101112",   # Border color
    "input_bg": "#282c34",  # Entry background
    "input_fg": "#e8e8e8",  # Entry foreground
    "label_fg": "#111315",  # Label color
    "error":   "#ff3333",   # Error red
    "success": "#0ba11d",   # Success green
    "warn":    "#ffb800",   # Warning yellow
    "info":    "#38c7ff",   # Info blue
    "disabled": "#454545",  # Disabled bg
}

def apply_modern_style(root):
    root.configure(bg=PALETTE["bg"])

    style = ttk.Style(root)
    style.theme_use("default")

    # Globale Einstellungen für ttk
    style.configure(".", background=PALETTE["bg"], foreground=PALETTE["fg"], font=("Segoe UI", 10))

    # Button-Style modern, flach, leicht abgerundet
    style.configure("TButton",
        background=PALETTE["accent"],
        foreground="#ffffff",
        borderwidth=0,
        padding=8,
        relief="flat",
        focusthickness=3,
        focuscolor=PALETTE["accent"],
        font=("Segoe UI", 10, "bold")
    )
    style.map("TButton",
        background=[("active", "#008fd4"), ("pressed", "#0099ff"), ("disabled", PALETTE["disabled"])],
        foreground=[("disabled", "#bbbbbb")]
    )

    # Entry modern
    style.configure("TEntry",
        foreground=PALETTE["input_fg"],
        fieldbackground=PALETTE["input_bg"],
        background=PALETTE["input_bg"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["accent"],
        borderwidth=1,
        padding=5,
        relief="flat"
    )

    # LabelFrame modern
    style.configure("TLabelframe",
        background=PALETTE["bg"],
        foreground=PALETTE["accent"],
        borderwidth=2,
        bordercolor=PALETTE["accent"]
    )
    style.configure("TLabelframe.Label",
        background=PALETTE["bg"],
        foreground=PALETTE["accent"],
        font=("Segoe UI", 11, "bold")
    )

    # Combobox
    style.configure("TCombobox",
        fieldbackground=PALETTE["input_bg"],
        background=PALETTE["input_bg"],
        foreground=PALETTE["input_fg"],
        borderwidth=1,
        relief="flat",
        arrowcolor=PALETTE["accent"],
        selectbackground=PALETTE["accent"],
        selectforeground="#ffffff"
    )

    # Checkbutton
    style.configure("TCheckbutton",
        background=PALETTE["bg"],
        foreground=PALETTE["fg"],
        font=("Segoe UI", 10),
        indicatorcolor=PALETTE["accent"],
        indicatorbackground=PALETTE["input_bg"]
    )
    style.map("TCheckbutton",
        background=[("active", PALETTE["input_bg"])],
        foreground=[("active", PALETTE["accent"])]
    )

    # Frame
    style.configure("TFrame",
        background=PALETTE["bg"]
    )

    # Progressbar
    style.configure("TProgressbar",
        background=PALETTE["accent"],
        troughcolor=PALETTE["input_bg"],
        borderwidth=0,
        thickness=8
    )


    # Individualfarben als Attribute verfügbar machen
    root.PALETTE = PALETTE

