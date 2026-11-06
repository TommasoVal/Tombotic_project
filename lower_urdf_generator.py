from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import math
import numpy as np


# ============================================================
# CONFIGURAZIONE SOGGETTO
# ============================================================
# Default anthropometric data for a subject. These values can be modified to generate different URDFs for various subjects.
SUBJECT = {
    # -------------------------
    # Dati globali
    # -------------------------
    "mass": 75.0,              # kg
    "height": 1.78,            # m

    # -------------------------
    # Misure antropometriche
    # -------------------------
    "pelvis_width": 0.20,      # distanza approssimata tra le anche
    "pelvis_depth": 0.16,
    "pelvis_height": 0.20,

    "thigh_length_R": 0.43,
    "thigh_length_L": 0.43,

    "shank_length_R": 0.43,
    "shank_length_L": 0.43,

    "foot_length_R": 0.26,
    "foot_length_L": 0.26,

    "foot_width_R": 0.10,
    "foot_width_L": 0.10,

    "foot_height": 0.07,
}


# ============================================================
# CONVENZIONE SISTEMA DI RIFERIMENTO
#
# X = medio-laterale
# Y = antero-posteriore
# Z = verticale
#
# L'asse positivo Z punta verso l'alto.
# ============================================================


# ============================================================
# ASSI ARTICOLARI
# ============================================================

JOINT_AXES = {

    # -------------------------
    # Anca
    # -------------------------
    "hip_R_flexion":  (1, 0, 0),
    "hip_R_adduction": (0, 1, 0),
    "hip_R_rotation":  (0, 0, 1),

    "hip_L_flexion":  (1, 0, 0),
    "hip_L_adduction": (0, 1, 0),
    "hip_L_rotation":  (0, 0, 1),

    # -------------------------
    # Ginocchio
    # -------------------------
    "knee_R": (1, 0, 0),
    "knee_L": (1, 0, 0),

    # -------------------------
    # Caviglia
    # -------------------------
    "ankle_R": (1, 0, 0),
    "ankle_L": (1, 0, 0),

    # Giunto subtalare semplificato
    "subtalar_R": (0, 1, 0),
    "subtalar_L": (0, 1, 0),
}


# ============================================================
# LIMITI ARTICOLARI
#
# Angoli in gradi.
# Verranno convertiti in radianti.
# ============================================================

JOINT_LIMITS_DEG = {

    # -------------------------
    # ANCA
    # -------------------------

    "hip_R_flexion": (-30.0, 120.0),
    "hip_R_adduction": (-35.0, 25.0),
    "hip_R_rotation": (-45.0, 45.0),

    "hip_L_flexion": (-30.0, 120.0),
    "hip_L_adduction": (-25.0, 35.0),
    "hip_L_rotation": (-45.0, 45.0),

    # -------------------------
    # GINOCCHIO
    # -------------------------

    "knee_R": (-5.0, 140.0),
    "knee_L": (-5.0, 140.0),

    # -------------------------
    # CAVIGLIA
    # -------------------------

    "ankle_R": (-30.0, 45.0),
    "ankle_L": (-30.0, 45.0),

    # -------------------------
    # SUBTALARE
    # -------------------------

    "subtalar_R": (-20.0, 20.0),
    "subtalar_L": (-20.0, 20.0),
}


# ============================================================
# MASSA SEGMENTALE
#
# Valori approssimativi antropometrici.
# Sono volutamente parametrizzati.
# ============================================================

def compute_segment_masses(total_mass):

    return {

        "pelvis":
            0.142 * total_mass,

        "thigh":
            0.100 * total_mass,

        "shank":
            0.0465 * total_mass,

        "foot":
            0.0145 * total_mass,
    }


# ============================================================
# CENTRI DI MASSA
#
# Frazione della lunghezza misurata dal segmento prossimale.
# ============================================================

COM_FRACTIONS = {

    "thigh": 0.433,
    "shank": 0.433,
    "foot": 0.500,
}


# ============================================================
# INERZIA DI UN CILINDRO / SEGMENTO
# ============================================================

def cylinder_inertia(mass, radius, length):

    i_long = 0.5 * mass * radius**2

    i_transverse = (
        mass / 12.0 *
        (3.0 * radius**2 + length**2)
    )

    return (
        i_transverse,
        i_transverse,
        i_long,
    )


# ============================================================
# FUNZIONE XML
# ============================================================

def add_inertial(
    link,
    mass,
    com,
    inertia,
):

    inertial = SubElement(
        link,
        "inertial"
    )

    SubElement(
        inertial,
        "origin",
        xyz=f"{com[0]:.6f} "
            f"{com[1]:.6f} "
            f"{com[2]:.6f}"
    )

    SubElement(
        inertial,
        "mass",
        value=f"{mass:.8f}"
    )

    ixx, iyy, izz = inertia

    SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.8e}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.8e}",
        iyz="0",
        izz=f"{izz:.8e}",
    )


# ============================================================
# GEOMETRIA CILINDRICA
# ============================================================

def add_cylinder_geometry(
    parent,
    radius,
    length,
):

    geometry = SubElement(
        parent,
        "geometry"
    )

    SubElement(
        geometry,
        "cylinder",
        radius=f"{radius:.6f}",
        length=f"{length:.6f}"
    )


# ============================================================
# LINK SEGMENTALE
# ============================================================

def add_segment_link(
    robot,
    name,
    mass,
    length,
    radius,
    com_fraction,
):

    link = SubElement(
        robot,
        "link",
        name=name
    )

    # --------------------------------------------------------
    # Centro di massa
    #
    # Il segmento viene rappresentato lungo Z negativo.
    # --------------------------------------------------------

    com_z = -length * com_fraction

    inertia = cylinder_inertia(
        mass,
        radius,
        length
    )

    add_inertial(
        link,
        mass,
        (0, 0, com_z),
        inertia
    )

    # --------------------------------------------------------
    # VISUAL
    # --------------------------------------------------------

    visual = SubElement(
        link,
        "visual"
    )

    SubElement(
        visual,
        "origin",
        xyz=f"0 0 {-length/2:.6f}"
    )

    add_cylinder_geometry(
        visual,
        radius,
        length
    )

    # --------------------------------------------------------
    # COLLISION
    # --------------------------------------------------------

    collision = SubElement(
        link,
        "collision"
    )

    SubElement(
        collision,
        "origin",
        xyz=f"0 0 {-length/2:.6f}"
    )

    add_cylinder_geometry(
        collision,
        radius,
        length
    )

    return name


# ============================================================
# PELVIS
# ============================================================

def add_pelvis(
    robot,
    mass,
    width,
    depth,
    height,
):

    link = SubElement(
        robot,
        "link",
        name="pelvis"
    )

    # Approssimazione ellissoidale tramite box
    ixx = mass / 12.0 * (
        depth**2 + height**2
    )

    iyy = mass / 12.0 * (
        width**2 + height**2
    )

    izz = mass / 12.0 * (
        width**2 + depth**2
    )

    add_inertial(
        link,
        mass,
        (0, 0, 0),
        (ixx, iyy, izz)
    )

    visual = SubElement(
        link,
        "visual"
    )

    geometry = SubElement(
        visual,
        "geometry"
    )

    SubElement(
        geometry,
        "box",
        size=f"{width:.6f} "
             f"{depth:.6f} "
             f"{height:.6f}"
    )

    collision = SubElement(
        link,
        "collision"
    )

    geometry = SubElement(
        collision,
        "geometry"
    )

    SubElement(
        geometry,
        "box",
        size=f"{width:.6f} "
             f"{depth:.6f} "
             f"{height:.6f}"
    )

    return link


# ============================================================
# JOINT
# ============================================================

def add_joint(
    robot,
    name,
    parent,
    child,
    joint_type,
    xyz,
    axis=None,
    limits=None,
):

    joint = SubElement(
        robot,
        "joint",
        name=name,
        type=joint_type
    )

    SubElement(
        joint,
        "parent",
        link=parent
    )

    SubElement(
        joint,
        "child",
        link=child
    )

    SubElement(
        joint,
        "origin",
        xyz=f"{xyz[0]:.6f} "
            f"{xyz[1]:.6f} "
            f"{xyz[2]:.6f}"
    )

    if joint_type in ("revolute", "continuous"):

        if axis is None:
            raise ValueError(
                f"Asse articolare mancante per {name}"
            )

        SubElement(
            joint,
            "axis",
            xyz=f"{axis[0]} "
                f"{axis[1]} "
                f"{axis[2]}"
        )

    if joint_type == "revolute" and limits:

        lower_deg, upper_deg = limits

        lower = math.radians(
            lower_deg
        )

        upper = math.radians(
            upper_deg
        )

        SubElement(
            joint,
            "limit",
            lower=f"{lower:.8f}",
            upper=f"{upper:.8f}",
            effort="1000",
            velocity="20"
        )

    return joint


# ============================================================
# GINOCCHIO
# ============================================================

def add_knee(
    robot,
    side,
    thigh_length,
    shank_length,
    thigh_link,
    shank_link,
):

    knee_name = f"knee_{side}"

    add_joint(
        robot,

        knee_name,

        thigh_link,
        shank_link,

        "revolute",

        (
            0,
            0,
            -thigh_length
        ),

        axis=JOINT_AXES[
            knee_name
        ],

        limits=JOINT_LIMITS_DEG[
            knee_name
        ],
    )


# ============================================================
# CAVIGLIA
# ============================================================

def add_ankle(
    robot,
    side,
    shank_length,
    shank_link,
    foot_link,
):

    ankle_name = f"ankle_{side}"

    add_joint(
        robot,

        ankle_name,

        shank_link,
        foot_link,

        "revolute",

        (
            0,
            0,
            -shank_length
        ),

        axis=JOINT_AXES[
            ankle_name
        ],

        limits=JOINT_LIMITS_DEG[
            ankle_name
        ],
    )


# ============================================================
# ANCA 3 DOF
#
# La catena è:
#
# pelvis
#    |
# hip_flexion
#    |
# hip_adduction
#    |
# hip_rotation
#    |
# thigh
#
# ============================================================

def add_hip(
    robot,
    side,
    pelvis_width,
    thigh_link,
):

    sign = -1 if side == "R" else 1

    x = sign * pelvis_width / 2.0

    # --------------------------------------------------------
    # Primo DOF: flesso-estensione
    # --------------------------------------------------------

    add_joint(
        robot,

        f"hip_{side}_flexion",

        "pelvis",

        f"hip_{side}_flexion_link",

        "revolute",

        (
            x,
            0,
            0
        ),

        axis=JOINT_AXES[
            f"hip_{side}_flexion"
        ],

        limits=JOINT_LIMITS_DEG[
            f"hip_{side}_flexion"
        ],
    )

    # --------------------------------------------------------
    # Secondo DOF: abduzione/adduzione
    # --------------------------------------------------------

    add_joint(
        robot,

        f"hip_{side}_adduction",

        f"hip_{side}_flexion_link",

        f"hip_{side}_adduction_link",

        "revolute",

        (0, 0, 0),

        axis=JOINT_AXES[
            f"hip_{side}_adduction"
        ],

        limits=JOINT_LIMITS_DEG[
            f"hip_{side}_adduction"
        ],
    )

    # --------------------------------------------------------
    # Terzo DOF: rotazione interna/esterna
    # --------------------------------------------------------

    add_joint(
        robot,

        f"hip_{side}_rotation",

        f"hip_{side}_adduction_link",

        thigh_link,

        "revolute",

        (0, 0, 0),

        axis=JOINT_AXES[
            f"hip_{side}_rotation"
        ],

        limits=JOINT_LIMITS_DEG[
            f"hip_{side}_rotation"
        ],
    )


# ============================================================
# GENERAZIONE DEI LINK VUOTI PER I DOF DELL'ANCA
# ============================================================

def add_intermediate_hip_links(
    robot,
    side
):

    for name in (
        f"hip_{side}_flexion_link",
        f"hip_{side}_adduction_link",
    ):

        SubElement(
            robot,
            "link",
            name=name
        )


# ============================================================
# GENERAZIONE COMPLETA
# ============================================================

def generate_lower_body_urdf(
    subject,
    output_file
):

    robot = Element(
        "robot",
        name="lower_body_biomechanical"
    )

    # ========================================================
    # MASSE
    # ========================================================

    masses = compute_segment_masses(
        subject["mass"]
    )

    # ========================================================
    # PELVIS
    # ========================================================

    add_pelvis(
        robot,

        masses["pelvis"],

        subject["pelvis_width"],
        subject["pelvis_depth"],
        subject["pelvis_height"]
    )

    # ========================================================
    # GAMBE
    # ========================================================

    for side in ("R", "L"):

        thigh_length = subject[
            f"thigh_length_{side}"
        ]

        shank_length = subject[
            f"shank_length_{side}"
        ]

        foot_length = subject[
            f"foot_length_{side}"
        ]

        foot_width = subject[
            f"foot_width_{side}"
        ]

        # ----------------------------------------------------
        # LINK INTERMEDI PER ANCA
        # ----------------------------------------------------

        add_intermediate_hip_links(
            robot,
            side
        )

        # ----------------------------------------------------
        # COSCIA
        # ----------------------------------------------------

        thigh = add_segment_link(
            robot,

            f"thigh_{side}",

            masses["thigh"],

            thigh_length,

            radius=0.065,

            com_fraction=COM_FRACTIONS[
                "thigh"
            ]
        )

        # ----------------------------------------------------
        # TIBIA
        # ----------------------------------------------------

        shank = add_segment_link(
            robot,

            f"shank_{side}",

            masses["shank"],

            shank_length,

            radius=0.045,

            com_fraction=COM_FRACTIONS[
                "shank"
            ]
        )

        # ----------------------------------------------------
        # PIEDE
        # ----------------------------------------------------

        foot = add_segment_link(
            robot,

            f"foot_{side}",

            masses["foot"],

            foot_length,

            radius=foot_width / 2.0,

            com_fraction=COM_FRACTIONS[
                "foot"
            ]
        )

        # ----------------------------------------------------
        # ANCA
        # ----------------------------------------------------

        add_hip(
            robot,

            side,

            subject["pelvis_width"],

            thigh
        )

        # ----------------------------------------------------
        # GINOCCHIO
        # ----------------------------------------------------

        add_knee(
            robot,

            side,

            thigh_length,

            shank_length,

            thigh,

            shank
        )

        # ----------------------------------------------------
        # CAVIGLIA
        # ----------------------------------------------------

        add_ankle(
            robot,

            side,

            shank_length,

            shank,

            foot
        )

    # ========================================================
    # SCRITTURA FILE
    # ========================================================
    save_urdf(robot, output_file)
    output_file = Path(
        output_file
    )

    print()
    print(
        "=========================================="
    )
    print(
        "URDF GENERATO"
    )
    print(
        "=========================================="
    )
    print(
        f"File: {output_file}"
    )
    print(
        f"Massa: {subject['mass']} kg"
    )
    print(
        f"Altezza: {subject['height']} m"
    )
    print(
        f"DOF articolari: {len(JOINT_AXES)-2}" # Exclude the subtalar joint from the count 
    )
    print(
        "=========================================="
    )

# ============================================================
# TEST E VALIDAZIONE CON PINOCCHIO
# ============================================================

def validate_pinocchio_model(urdf_path):
    model = pin.buildModelFromUrdf(str(urdf_path))

    print("\n=== PINOCCHIO MODEL ===")
    print(f"nq = {model.nq}")
    print(f"nv = {model.nv}")
    print(f"njoints = {model.njoints}")

    total_mass = 0.0

    for i, name in enumerate(model.names):
        mass = model.inertias[i].mass
        total_mass += mass
        print(f"{i:2d}  {name:30s} mass={mass:.6f} kg")

    print(f"\nTotal mass: {total_mass:.6f} kg")

    return model

# ============================================================
# SAVE URDF
# ============================================================

def save_urdf(robot, output_path):
    """
    Salva l'URDF in formato XML leggibile e indentato.
    """

    # Pretty-print XML
    ET.indent(robot, space="  ", level=0)

    tree = ET.ElementTree(robot)

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )

    print(f"URDF salvato in: {output_path}")

# ============================================================
# VISUALIZZAZIONE
# ============================================================

def visualize_model(model, urdf_path):
    """
    Visualizza il modello URDF letto da Pinocchio.

    Non modifica il modello dinamico.
    Mostra semplicemente la configurazione q nel visualizzatore.
    """

    print("\n=== AVVIO VISUALIZZAZIONE ===")

    # Modello geometrico VISUAL
    visual_model = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.VISUAL
    )

    # Modello geometrico COLLISION
    collision_model = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.COLLISION
    )

    print("Visual geometry:", visual_model.ngeoms)
    print("Collision geometry:", collision_model.ngeoms)

    # Crea visualizzatore
    viz = MeshcatVisualizer(
        model,
        collision_model,
        visual_model
    )

    # Avvia Meshcat
    # viz.initViewer(open=True)
    viz.initViewer(open=False)
    print("\nMeshcat:")
    print(viz.viewer.url())

    # Carica il modello
    viz.loadViewerModel()

    # Configurazione neutra di Pinocchio
    #q0 = pin.randomConfiguration(model)
    q0 = pin.neutral(model)
    q0 = np.array([np.pi/2, 0.0, 0.0, -np.pi/2, np.pi/2, np.pi/2, 0.0, 0.0, -np.pi/2, np.pi/2])
      

    print("\nConfigurazione neutra:")
    print(q0)

    # Visualizza q0
    viz.display(q0)

    print("\n=== VISUALIZZAZIONE PRONTA ===")
    print("Controlla la finestra/browser di Meshcat.")

    return viz



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # generate_lower_body_urdf(
    #     SUBJECT,
    #     "model/lower_body.urdf"
    # )

    # Validate the generated URDF with Pinocchio
    model =validate_pinocchio_model("model/lower_body.urdf")

    viz = visualize_model(model,"model/lower_body.urdf")

    # Se chiudi run, non è possibile visualizzare più il mesh
    input("\nPremi INVIO per chiudere...")