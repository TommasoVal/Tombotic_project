
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

URDF_PATH = Path("lower_model/lower_body.urdf")

EXPECTED_DOF = 10

ANATOMICAL_LINKS = [
    "pelvis",
    "thigh_L",
    "shank_L",
    "foot_L",
    "thigh_R",
    "shank_R",
    "foot_R",
]

EXPECTED_JOINTS = [
    "hip_L_flexion",
    "hip_L_adduction",
    "hip_L_rotation",
    "knee_L",
    "ankle_L",
    "hip_R_flexion",
    "hip_R_adduction",
    "hip_R_rotation",
    "knee_R",
    "ankle_R",
]


# ============================================================
# UTILITY XML
# ============================================================

def parse_vector(text, expected_size=3):
    """Converte una stringa XML 'x y z' in un array NumPy."""
    if text is None:
        return np.zeros(expected_size, dtype=float)

    values = np.fromstring(text, sep=" ")

    if values.size != expected_size:
        raise ValueError(
            f"Vettore non valido: {text!r}. "
            f"Attesi {expected_size} valori."
        )

    return values


def parse_origin(element):
    """
    Legge un elemento <origin> URDF.

    Restituisce:
        xyz: traslazione [m]
        rpy: roll, pitch, yaw [rad]
    """
    if element is None:
        return {
            "xyz": np.zeros(3),
            "rpy": np.zeros(3),
        }

    return {
        "xyz": parse_vector(element.get("xyz")),
        "rpy": parse_vector(element.get("rpy")),
    }


# ============================================================
# PARSER DEI LINK
# ============================================================

def parse_link(link_element):
    """Estrae proprietà inerziali e nome di un link."""

    name = link_element.get("name")

    inertial = link_element.find("inertial")

    # Link puramente cinematici: massa e inerzia nulle.
    if inertial is None:
        return {
            "name": name,
            "mass": 0.0,
            "com": np.zeros(3),
            "inertia": np.zeros((3, 3)),
            "inertial_origin": parse_origin(None),
            "has_inertial": False,
        }

    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    origin_element = inertial.find("origin")

    if mass_element is None or inertia_element is None:
        raise ValueError(
            f"Link {name!r}: elemento inertial incompleto."
        )

    mass = float(mass_element.get("value"))

    if mass < 0:
        raise ValueError(
            f"Link {name!r}: massa negativa."
        )

    # Il tensore è espresso nel frame inertial,
    # non necessariamente nel frame del link.
    I = inertia_element

    inertia = np.array([
        [
            float(I.get("ixx")),
            float(I.get("ixy")),
            float(I.get("ixz")),
        ],
        [
            float(I.get("ixy")),
            float(I.get("iyy")),
            float(I.get("iyz")),
        ],
        [
            float(I.get("ixz")),
            float(I.get("iyz")),
            float(I.get("izz")),
        ],
    ])

    inertial_origin = parse_origin(origin_element)

    return {
        "name": name,
        "mass": mass,
        "com": inertial_origin["xyz"].copy(),
        "inertia": inertia,
        "inertial_origin": inertial_origin,
        "has_inertial": True,
    }


# ============================================================
# PARSER DEI GIUNTI
# ============================================================

def parse_joint(joint_element):
    """Estrae le informazioni cinematiche di un giunto."""

    name = joint_element.get("name")
    joint_type = joint_element.get("type")

    parent_element = joint_element.find("parent")
    child_element = joint_element.find("child")

    if parent_element is None or child_element is None:
        raise ValueError(
            f"Joint {name!r}: parent o child mancante."
        )

    parent = parent_element.get("link")
    child = child_element.get("link")

    origin = parse_origin(joint_element.find("origin"))

    axis_element = joint_element.find("axis")

    if axis_element is None:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = parse_vector(axis_element.get("xyz"))

    # In URDF, il vettore axis deve essere unitario
    # per poter essere utilizzato direttamente.
    if joint_type in ("revolute", "continuous", "prismatic"):
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            raise ValueError(
                f"Joint {name!r}: asse nullo."
            )

        axis = axis / axis_norm

    limit_element = joint_element.find("limit")

    limits = None

    if limit_element is not None:
        lower = limit_element.get("lower")
        upper = limit_element.get("upper")

        limits = {
            "lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None,
            "effort": (
                float(limit_element.get("effort"))
                if limit_element.get("effort") is not None
                else None
            ),
            "velocity": (
                float(limit_element.get("velocity"))
                if limit_element.get("velocity") is not None
                else None
            ),
        }

    return {
        "name": name,
        "type": joint_type,
        "parent": parent,
        "child": child,
        "origin": origin,
        "axis": axis,
        "limits": limits,
    }


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

def load_urdf(urdf_path):
    """Legge l'URDF e costruisce il modello dati."""

    urdf_path = Path(urdf_path)

    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"URDF non trovato: {urdf_path.resolve()}"
        )

    root = ET.parse(urdf_path).getroot()

    if root.tag != "robot":
        raise ValueError("Il file non contiene un elemento <robot>.")

    links = {}
    joints = {}

    for element in root.findall("link"):
        link = parse_link(element)
        name = link["name"]

        if name in links:
            raise ValueError(f"Link duplicato: {name}")

        links[name] = link

    for element in root.findall("joint"):
        joint = parse_joint(element)
        name = joint["name"]

        if name in joints:
            raise ValueError(f"Joint duplicato: {name}")

        joints[name] = joint

    return {
        "name": root.get("name", "unnamed_robot"),
        "links": links,
        "joints": joints,
    }


# ============================================================
# VALIDAZIONE STRUTTURA
# ============================================================

def validate_model(model):
    """Esegue controlli strutturali indipendenti da Pinocchio."""

    links = model["links"]
    joints = model["joints"]

    errors = []

    print("\n=== VALIDAZIONE STRUTTURA ===")

    # Link attesi
    for name in ANATOMICAL_LINKS:
        if name not in links:
            errors.append(f"Link anatomico mancante: {name}")

    # Joint attesi
    for name in EXPECTED_JOINTS:
        if name not in joints:
            errors.append(f"Joint mancante: {name}")

    # Ogni joint deve collegare link esistenti
    for joint in joints.values():
        if joint["parent"] not in links:
            errors.append(
                f"{joint['name']}: parent inesistente "
                f"{joint['parent']!r}"
            )

        if joint["child"] not in links:
            errors.append(
                f"{joint['name']}: child inesistente "
                f"{joint['child']!r}"
            )

    # Ogni link figlio può avere un solo joint padre.
    child_links = [
        joint["child"] for joint in joints.values()
    ]

    if len(child_links) != len(set(child_links)):
        errors.append(
            "Uno o più link hanno più giunti genitori."
        )

    # Un solo link radice.
    child_set = set(child_links)

    roots = [
        name for name in links
        if name not in child_set
    ]

    if len(roots) != 1:
        errors.append(
            f"Attesa una sola radice; trovate: {roots}"
        )
    else:
        print(f"Link radice: {roots[0]}")

    # Nel modello attuale tutti i joint sono rotazionali.
    movable_types = ("revolute", "continuous", "prismatic")

    dof = sum(
        joint["type"] in movable_types
        for joint in joints.values()
    )

    print(f"Link totali: {len(links)}")
    print(f"Joint totali: {len(joints)}")
    print(f"DOF scalari attesi: {dof}")

    if dof != EXPECTED_DOF:
        errors.append(
            f"DOF attesi {EXPECTED_DOF}, trovati {dof}."
        )

    # Verifica raggiungibilità di ogni link dalla radice.
    if len(roots) == 1:
        visited = {roots[0]}
        changed = True

        while changed:
            changed = False

            for joint in joints.values():
                if (
                    joint["parent"] in visited
                    and joint["child"] not in visited
                ):
                    visited.add(joint["child"])
                    changed = True

        unreachable = set(links) - visited

        if unreachable:
            errors.append(
                f"Link non raggiungibili dalla radice: "
                f"{sorted(unreachable)}"
            )

    if errors:
        for message in errors:
            print(f"[FAIL] {message}")
    else:
        print("[OK] Struttura coerente.")

    return errors


# ============================================================
# REPORT
# ============================================================

def print_report(model):
    """Stampa riepilogo di masse, inerzie e giunti."""

    print("\n" + "=" * 65)
    print(f"MODELLO: {model['name']}")
    print("=" * 65)

    print("\n--- LINK ---")

    total_mass = 0.0

    for link in model["links"].values():
        mass = link["mass"]
        total_mass += mass

        print(
            f"{link['name']:25s} "
            f"massa={mass:8.4f} kg "
            f"inerziale={'sì' if link['has_inertial'] else 'no'}"
        )

        if link["has_inertial"]:
            print(f"  COM locale: {link['com']}")
            print(f"  Inerzia locale:\n{link['inertia']}")

    print(f"\nMassa totale: {total_mass:.4f} kg")

    print("\n--- JOINT ---")

    for joint in model["joints"].values():
        print(
            f"{joint['name']:25s} "
            f"{joint['type']:10s} "
            f"{joint['parent']} -> {joint['child']}"
        )

        print(f"  xyz:  {joint['origin']['xyz']}")
        print(f"  rpy:  {joint['origin']['rpy']}")
        print(f"  asse: {joint['axis']}")

        if joint["limits"] is not None:
            print(f"  limiti: {joint['limits']}")


# ============================================================
# MAIN
# ============================================================

def main():
    model = load_urdf(URDF_PATH)

    print_report(model)

    errors = validate_model(model)

    if errors:
        raise SystemExit(
            f"\nValidazione fallita: {len(errors)} problemi."
        )

    print("\n[OK] Parser URDF completato.")
    print("Nessuna funzione Pinocchio è stata utilizzata.")


if __name__ == "__main__":
    main()