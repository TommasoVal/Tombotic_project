"""Plot a Cinematica JSON export using the existing TestEnvironment canvas.

Usage: .venv/Scripts/python.exe tools/plot_kinematic_json.py [file.json]
Add --body-part 4 for the right lower leg, or --check to validate without GUI.
"""

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (
    ("Giroscopio", ("gx", "gy", "gz")),
    ("Accelerazione lineare locale", ("ax", "ay", "az")),
    ("Accelerazione lineare globale", ("agx", "agy", "agz")),
)


def load_data(path, body_part=None):
    """Read and validate plotted channels without changing or resampling data."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("format") != "kinelab_kinematic_preview" or data.get("version") != 1:
        raise ValueError("Atteso un export kinelab_kinematic_preview versione 1.")
    devices = [device for device in data["devices"] if body_part is None
               or device["configuration"]["body_part_id"] == body_part]
    if not devices:
        raise ValueError("Nessun segmento corrisponde alla selezione.")
    for device in devices:
        samples = device["samples"]
        if len(samples) < 2:
            raise ValueError("Servono almeno due campioni per segmento.")
        previous = -math.inf
        for sample in samples:
            for key in ("t", *(key for _, keys in SIGNALS for key in keys)):
                value = sample[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Canale {key}: atteso un numero finito.")
            if sample["t"] <= previous:
                raise ValueError("I timestamp devono essere strettamente crescenti.")
            previous = sample["t"]
        if samples[0]["t"] != 0:
            raise ValueError("L'export Cinematica deve iniziare a t=0 s.")
    for _, keys in SIGNALS:
        metadata = data["channels"]["/".join(keys)]
        if not metadata.get("unit") or not metadata.get("frame"):
            raise ValueError("Unita o frame mancanti nei metadati dei canali.")
    return data, devices


def show_plots(data, devices, filename):
    # Reuse the application plotter; no broker, DB or full MainWindow startup.
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtWidgets import QApplication, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget
    from gui.kinematics_tab import KinematicCanvas

    app = QApplication([])
    window = QTabWidget()
    window.setWindowTitle(f"Cinematica JSON - {filename}")
    window.resize(1050, 930)
    for device in devices:
        config = device["configuration"]
        samples = device["samples"]
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(
            f"Device {config['id']} | body_part_id {config['body_part_id']} | "
            f"{len(samples)} campioni | durata {samples[-1]['t']:.3f} s"
        ))
        for title, keys in SIGNALS:
            metadata = data["channels"]["/".join(keys)]
            unit = metadata["unit"]
            gravity = " | gravity-free" if metadata.get("gravity_removed") else ""
            layout.addWidget(QLabel(f"{title} | {metadata['frame']} | {unit}{gravity}"))
            canvas = KinematicCanvas(
                lambda s=samples, k=keys, u=unit: (s, k, k, u)
            )
            layout.addWidget(canvas)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        window.addTab(scroll, f"{config['name']} (Device {config['id']})")
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description="Plotta gyro e accelerazione del JSON Cinematica.")
    parser.add_argument("file", nargs="?", type=Path, default=ROOT / "cinematica_simulata.json")
    parser.add_argument("--body-part", type=int, help="Filtra per body_part_id (4 = gamba destra).")
    parser.add_argument("--check", action="store_true", help="Verifica il file senza aprire finestre.")
    args = parser.parse_args()
    try:
        data, devices = load_data(args.file, args.body_part)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Errore lettura JSON: {exc}\n")
    for device in devices:
        config = device["configuration"]
        print(f"Device {config['id']} / parte {config['body_part_id']}: "
             f"{len(device['samples'])} campioni")
        print(data)
    return 0 if args.check else show_plots(data, devices, args.file.name)


if __name__ == "__main__":
    sys.exit(main())
