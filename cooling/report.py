import io
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


def _plot_severity(feats: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(6, 2.2))
    plt.plot(feats["t_end"], feats["severity"])
    plt.xlabel("Time (s)")
    plt.ylabel("Severity (0-100)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_report(
    feats: pd.DataFrame, signals: pd.DataFrame, green_hi: int, yellow_hi: int
) -> bytes:
    sev_png = _plot_severity(feats)
    worst = float(feats["severity"].max())
    avg = float(feats["severity"].mean())
    n_red = int((feats["severity"] >= yellow_hi).sum())
    n_yellow = int(
        ((feats["severity"] >= green_hi) & (feats["severity"] < yellow_hi)).sum()
    )

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, H - 2 * cm, "PC Cooling Predictive Maintenance — Report")

    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, H - 3 * cm, f"Worst severity: {int(worst)} / 100")
    c.drawString(2 * cm, H - 3.7 * cm, f"Average severity: {int(avg)} / 100")
    c.drawString(
        2 * cm,
        H - 4.4 * cm,
        f"Yellow threshold: < {yellow_hi},  Green threshold: < {green_hi}",
    )
    c.drawString(
        2 * cm, H - 5.1 * cm, f"Yellow windows: {n_yellow}     Red windows: {n_red}"
    )

    img = ImageReader(io.BytesIO(sev_png))
    c.drawImage(
        img,
        2 * cm,
        H - 10 * cm,
        width=16 * cm,
        height=4 * cm,
        preserveAspectRatio=True,
        mask="auto",
    )

    c.setFont("Helvetica", 10)
    c.drawString(
        2 * cm,
        H - 11 * cm,
        "Signals (last snapshot): CPU/GPU temps (°C), fan RPM, CPU load (%)",
    )
    last = signals.iloc[-1]
    cpu = last.get("cpu_temp", float("nan"))
    gpu = last.get("gpu_temp", float("nan"))
    rpm = last.get("fan_rpm", float("nan"))
    load = last.get("cpu_load", float("nan"))
    amb = last.get("ambient", float("nan"))
    c.drawString(
        2 * cm,
        H - 11.7 * cm,
        f"CPU: {cpu:.1f} °C   GPU: {gpu:.1f} °C   Fan: {rpm:.0f} RPM   Load: {load:.0f}%   Ambient: {amb:.1f} °C",
    )

    c.showPage()
    c.save()
    out.seek(0)
    return out.getvalue()
