"""Structured logging configuration."""
from __future__ import annotations

import logging
import os
import sys

# Logging'in birden fazla kez kurulmasini engelleyen modul seviyesinde bayrak.
# basicConfig yalnizca bir kez etki ettigi icin idempotent kurulum sagliyoruz.
_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    # Kok logger'i tek seferlik yapilandirir. Birden cok modul get_logger
    # cagirdiginda tekrar tekrar calismamasi icin _CONFIGURED ile korunur.
    global _CONFIGURED
    # Zaten kuruldiysa erken don; aksi halde basicConfig cift cagrilir ve
    # handler'lar yinelenip loglar coklanir.
    if _CONFIGURED:
        return
    # Seviye oncelik sirasi: fonksiyon argumani > LOG_LEVEL env > INFO varsayilani.
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # Konteyner ortaminda loglarin "docker logs" ile gorunmesi icin
        # stderr yerine stdout'a yaziyoruz.
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    # Modullerin kullanacagi standart giris noktasi: logger'i isim ile dondurur.
    # Ilk cagri kurulumu garanti altina alir, boylece her modulun ayrica
    # configure_logging cagirmasina gerek kalmaz.
    configure_logging()
    return logging.getLogger(name)
