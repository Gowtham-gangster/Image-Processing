"""
alert_manager.py
================
Real-Time Alert Dispatcher for Mask-Aware Surveillance System.

Supports three notification channels:
  - Slack  (Incoming Webhook)
  - Email  (SMTP, e.g. Gmail / Outlook)
  - Webhook (Generic HTTP POST)

Configuration
-------------
Channels are configured via ``alerts_config.json`` at the project root.
Example:

    {
        "slack_webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
        "email": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "sender": "system@example.com",
            "password": "app-password",
            "recipients": ["security@example.com"]
        },
        "webhook_url": "https://my-siem.example.com/ingest"
    }

Usage
-----
    from alert_manager import AlertManager
    am = AlertManager()
    am.send_alert("unknown_person", camera_id="Cam_01", confidence=0.45)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import urllib.request
import urllib.error

from config import ALERT_CONFIG_PATH, LOG_LEVEL
from database import DatabaseManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ── Alert types ──────────────────────────────────────────────────────────────
ALERT_UNKNOWN_PERSON = "unknown_person"
ALERT_UNMASKED       = "unmasked_person"
ALERT_SPOOF          = "spoof_attempt"

ALERT_LABELS = {
    ALERT_UNKNOWN_PERSON: "🚨 Unknown Person Detected",
    ALERT_UNMASKED:       "⚠️  Unmasked Person Detected",
    ALERT_SPOOF:          "🎭 Spoof Attempt Detected",
}


class AlertManager:
    """
    Dispatches alerts to configured channels when a surveillance event occurs.
    All sends are executed in daemon threads to avoid blocking the video loop.
    """

    def __init__(self, config_path: str = ALERT_CONFIG_PATH, db: Optional[DatabaseManager] = None):
        self._config: dict = {}
        self._config_path = config_path
        self.db = db or DatabaseManager()
        self._load_config()
        logger.info("AlertManager ready. Channels: %s", self._active_channels())

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info("Alert config loaded from %s", self._config_path)
            except Exception as e:
                logger.warning("Failed to load alert config: %s", e)
                self._config = {}
        else:
            self._config = {}
            logger.info(
                "No alert config found at %s — alerts disabled. "
                "POST to /alerts/config to configure.",
                self._config_path,
            )

    def save_config(self, new_config: dict) -> None:
        """Persist a new config dict to disk and reload."""
        self._config = new_config
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2)
            logger.info("Alert config saved.")
        except Exception as e:
            logger.error("Failed to save alert config: %s", e)

    def get_config(self) -> dict:
        return dict(self._config)

    def _active_channels(self) -> list[str]:
        channels = []
        if self._config.get("slack_webhook_url"):
            channels.append("Slack")
        if self._config.get("email", {}).get("smtp_host"):
            channels.append("Email")
        if self._config.get("webhook_url"):
            channels.append("Webhook")
        return channels or ["none"]

    # ── Public API ────────────────────────────────────────────────────────────

    def send_alert(
        self,
        alert_type: str,
        camera_id: str = "Unknown",
        person_id: str = "Unknown",
        confidence: float = 0.0,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Fire an alert asynchronously on all configured channels.

        Parameters
        ----------
        alert_type  : One of ALERT_UNKNOWN_PERSON, ALERT_UNMASKED, ALERT_SPOOF.
        camera_id   : Camera identifier string.
        person_id   : Person identifier or 'Unknown Person'.
        confidence  : Recognition confidence / similarity.
        extra       : Optional extra metadata dict included in payloads.
        """
        if not self._active_channels() or self._active_channels() == ["none"]:
            logger.debug("No alert channels configured. Skipping alert.")
            return

        payload = {
            "alert_type":  alert_type,
            "label":       ALERT_LABELS.get(alert_type, alert_type),
            "camera_id":   camera_id,
            "person_id":   person_id,
            "confidence":  round(confidence, 3),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            **(extra or {}),
        }

        # Log safely into Database natively on main thread or background? Doing it on main thread before dispatch ensures reliable audit trails.
        try:
            self.db.add_alert(
                alert_type=payload["alert_type"],
                camera_id=payload["camera_id"],
                person_id=payload["person_id"],
                confidence=payload["confidence"],
                timestamp=payload["timestamp"]
            )
        except Exception as e:
            logger.error("Failed persisting alert to DB: %s", e)

        # Dispatch non-blocking
        t = threading.Thread(target=self._dispatch_all, args=(payload,), daemon=True)
        t.start()

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _dispatch_all(self, payload: dict) -> None:
        if self._config.get("slack_webhook_url"):
            self._send_slack(payload)
        if self._config.get("email", {}).get("smtp_host"):
            self._send_email(payload)
        if self._config.get("webhook_url"):
            self._send_webhook(payload)

    def _send_slack(self, payload: dict) -> None:
        url = self._config["slack_webhook_url"]
        label = payload["label"]
        ts = payload["timestamp"]

        body = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": label},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Camera:*\n{payload['camera_id']}"},
                        {"type": "mrkdwn", "text": f"*Person ID:*\n{payload['person_id']}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{payload['confidence']:.1%}"},
                        {"type": "mrkdwn", "text": f"*Time (UTC):*\n{ts}"},
                    ],
                },
                {"type": "divider"},
            ]
        }

        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Slack alert sent: %s", label)
        except Exception as e:
            logger.error("Slack alert failed: %s", e)

    def _send_email(self, payload: dict) -> None:
        cfg = self._config.get("email", {})
        host = cfg.get("smtp_host", "smtp.gmail.com")
        port = cfg.get("smtp_port", 587)
        sender = cfg.get("sender", "")
        password = cfg.get("password", "")
        recipients = cfg.get("recipients", [])

        if not recipients:
            return

        subject = f"[MaskAwareID] {payload['label']} — {payload['camera_id']}"
        body_html = f"""
        <html><body style="font-family:sans-serif;background:#0f0f12;color:#e0e0e0;padding:20px">
        <h2 style="color:#ef4444">{payload['label']}</h2>
        <table>
          <tr><td><b>Camera</b></td><td>{payload['camera_id']}</td></tr>
          <tr><td><b>Person ID</b></td><td>{payload['person_id']}</td></tr>
          <tr><td><b>Confidence</b></td><td>{payload['confidence']:.1%}</td></tr>
          <tr><td><b>Time (UTC)</b></td><td>{payload['timestamp']}</td></tr>
        </table>
        </body></html>
        """

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())

            logger.info("Email alert sent to %s", recipients)
        except Exception as e:
            logger.error("Email alert failed: %s", e)

    def _send_webhook(self, payload: dict) -> None:
        url = self._config["webhook_url"]
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Webhook alert sent to %s", url)
        except Exception as e:
            logger.error("Webhook alert failed: %s", e)
