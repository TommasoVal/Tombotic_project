from xml.parsers.expat import model

import numpy as np
import pinocchio as pin

from urdf_parser import load_urdf

# ============================================================
# ORDINE DEI GIUNTI MOBILI PER ESSERE COMPATIBILE CON PINOCCHIO
# ============================================================
JOINT_ORDER = [
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
# TRASFORMAZIONI RIGIDE
# ============================================================

def skew(a):
    """Matrice antisimmetrica associata al vettore 3D."""
    x, y, z = np.asarray(a, dtype=float)

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def rotation_rpy(rpy):
    """
    Rotazione URDF da roll, pitch, yaw.
    Convenzione: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll, pitch, yaw = np.asarray(rpy, dtype=float)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])


    # # Supponiamo che A e B siano le tue matrici
    # print("Dimensione Rz:", Rz.shape)
    # print("Dimensione Ry:", Ry.shape)
    # print("Dimensione Rx:", Ry.shape)

    # try:
    #     # Esegue il prodotto matriciale (usa @ oppure np.dot)
    #     risultato = Rz @ Ry @ Rx
    #     print("Prodotto calcolato con successo!")
        
    #     # Per verificare se è l'identità tollerando i piccoli errori decimali:
    #     identita_teorica = np.eye(risultato.shape[0])
    #     if np.allclose(risultato, identita_teorica):
    #         print("È una matrice identità!")
    #     else:
    #         print("Il risultato non è un'identità perfetta. Ecco il risultato:\n", risultato)

    # except ValueError as e:
    #     print("Errore di dimensioni:", e)
    # except MemoryError:
    #     print("Errore: Memoria RAM esaurita! Le matrici sono troppo grandi.")

    # la chiocciola fa la moltiplicazione di matrici
    return Rz @ Ry @ Rx


def transform_from_origin(origin):
    """Costruisce T 4x4 da xyz e rpy del parser URDF."""
    T = np.eye(4)
    T[:3, :3] = rotation_rpy(origin["rpy"])
    T[:3, 3] = origin["xyz"]
    return T


def rotation_axis_angle(axis, angle):
    """Rotazione attorno a un asse espresso nel frame locale."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)

    if norm < 1e-12:
        raise ValueError("Asse di rotazione nullo.")

    axis = axis / norm
    K = skew(axis)

    return (
        np.eye(3)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )


def joint_motion_transform(joint, value=0.0):
    """
    Trasformazione dovuta al movimento del giunto.
    value è in radianti per revolute/continuous,
    metri per prismatic, ignorato per fixed.
    """
    T = np.eye(4)
    joint_type = joint["type"]

    if joint_type in ("revolute", "continuous"):
        T[:3, :3] = rotation_axis_angle(
            joint["axis"], value
        )

    elif joint_type == "prismatic":
        T[:3, 3] = joint["axis"] * value

    elif joint_type == "fixed":
        pass

    else:
        raise NotImplementedError(
            f"Tipo di giunto non supportato: {joint_type}"
        )

    return T


# ============================================================
# ORDINE TOPOLOGICO
# ============================================================

def build_joint_tree(model):
    """
    Costruisce l'elenco dei figli di ciascun link.
    Mantiene l'ordine necessario a propagare le trasformazioni
    dalla radice verso i link terminali.
    """
    children = {name: [] for name in model["links"]}

    for joint in model["joints"].values():
        children[joint["parent"]].append(joint)

    child_links = {
        joint["child"] for joint in model["joints"].values()
    }

    roots = [
        name for name in model["links"]
        if name not in child_links
    ]

    if len(roots) != 1:
        raise ValueError(
            f"Attesa una sola radice, trovate: {roots}"
        )

    return roots[0], children


# ============================================================
# CINEMATICA DIRETTA
# ============================================================

def forward_kinematics(model, q, T_world_root=None):
    """
    Calcola la posa globale di ogni link(Le matrici di trasformazione omogenea Tij).

    Parametri
    ---------
    model:
        Dizionario prodotto da load_urdf().
    q:
        Dizionario {nome_giunto: valore}.
        Angoli in radianti, traslazioni in metri.
        I giunti non specificati assumono valore zero.
    T_world_root:
        Trasformazione globale della radice.
        Se omessa, la radice coincide con il mondo.

    Restituisce
    -----------
    poses:
        Dizionario {nome_link: matrice 4x4}.
    """
    q = {} if q is None else q

    root, children = build_joint_tree(model)

    if T_world_root is None:
        T_world_root = np.eye(4)

    T_world_root = np.asarray(T_world_root, dtype=float)

    if T_world_root.shape != (4, 4):
        raise ValueError("T_world_root deve essere 4x4.")

    poses = {root: T_world_root.copy()}
    stack = [root]

    while stack:
        parent_name = stack.pop()
        T_world_parent = poses[parent_name]

        for joint in children[parent_name]:
            name = joint["name"]
            child_name = joint["child"]

            # Valore della rotazione locale del giunto per adesso a 0.0
            value = q[name] if name in q else 0.0
            

            T_parent_joint = transform_from_origin(
                joint["origin"]
            )

            T_joint_child = joint_motion_transform(
                joint, value
            )

            # Questa da l'effettiva T0i*Tij, con i padre e j figlio attraverso il giunto nel mezzo Tij = Ti_joint*Tjoint_j
            T_world_child = (
                T_world_parent
                @ T_parent_joint
                @ T_joint_child
            )
            # poses tratta il child come se fosse il link finale
            poses[child_name] = T_world_child
            stack.append(child_name)

    if len(poses) != len(model["links"]):
        missing = set(model["links"]) - set(poses)
        raise ValueError(
            f"Link non raggiunti dalla radice: {sorted(missing)}"
        )

    return poses


# ============================================================
# INFORMAZIONI DEI GIUNTI NEL MONDO
# ============================================================

def joint_world_frames(model, poses):
    """
    Per ogni giunto restituisce:
      - origine del giunto nel mondo;
      - asse del giunto nel mondo.

    L'asse è espresso nel frame del giunto dopo aver applicato
    l'origin del giunto e prima della rotazione articolare.
    """
    result = {}

    for joint in model["joints"].values():
        parent_name = joint["parent"]

        T_world_parent = poses[parent_name]
        T_parent_joint = transform_from_origin(
            joint["origin"]
        )

        T_world_joint = T_world_parent @ T_parent_joint

        axis_local = joint["axis"]
        axis_world = (
            T_world_joint[:3, :3] @ axis_local
        )

        result[joint["name"]] = {
            "position": T_world_joint[:3, 3].copy(),
            "axis": axis_world,
            "rotation": T_world_joint[:3, :3].copy(),
            "joint_type": joint["type"],
            "parent": joint["parent"],
            "child": joint["child"],
        }

    return result


# ============================================================
# JACOBIANO GEOMETRICO DI UN PUNTO
# ============================================================

def geometric_jacobian(model, q, link_name, point_local=None):
    """
    Jacobiano 6 x n per un punto solidale a un link.

    Convenzione delle righe:
        [vx, vy, vz, wx, wy, wz]

    point_local è espresso nel frame del link.
    Per default si usa l'origine del link.

    Le colonne seguono l'ordine dei giunti mobili
    nel dizionario model["joints"].
    """
    if link_name not in model["links"]:
        raise ValueError(f"Link sconosciuto: {link_name}")

    poses = forward_kinematics(model, q)
    joint_frames = joint_world_frames(model, poses)

    if point_local is None:
        point_local = np.zeros(3)

    point_local = np.asarray(point_local, dtype=float)

    T_world_link = poses[link_name]

    point_world = (
        T_world_link[:3, :3] @ point_local
        + T_world_link[:3, 3]
    )

    # Catena di antenati: dal link al root.
    joint_by_child = {
        joint["child"]: joint
        for joint in model["joints"].values()
    }

    ancestors = []
    current = link_name

    while current in joint_by_child:
        joint = joint_by_child[current]
        ancestors.append(joint)
        current = joint["parent"]

    ancestors.reverse()

    movable_joints = [
        model["joints"][name]
        for name in JOINT_ORDER
    ]

    J = np.zeros((6, len(movable_joints)))

    column_by_name = {
        joint["name"]: i
        for i, joint in enumerate(movable_joints)
    }

    for joint in ancestors:
        if joint["type"] == "fixed":
            continue

        col = column_by_name[joint["name"]]
        frame = joint_frames[joint["name"]]

        axis = frame["axis"]
        origin = frame["position"]

        if joint["type"] in ("revolute", "continuous"):
            J[:3, col] = np.cross(
                axis, point_world - origin
            )
            J[3:, col] = axis

        elif joint["type"] == "prismatic":
            J[:3, col] = axis
            J[3:, col] = 0.0

        # print("\n", joint["name"])
        # print("origin:", frame["position"])
        # print("axis:  ", frame["axis"])

    return J


# ============================================================
# CENTRI DI MASSA
# ============================================================

def center_of_mass_positions(model, poses):
    """
    Restituisce la posizione globale del COM di ogni link
    che possiede un elemento inertial.
    """
    com_positions = {}

    for name, link in model["links"].items():
        if not link["has_inertial"] :
            continue

        T_world_link = poses[name]

        com_local = link["com"]

        com_world = (
            T_world_link[:3, :3] @ com_local
            + T_world_link[:3, 3]
        )

        com_positions[name] = com_world

    return com_positions


def total_center_of_mass(model, poses):
    """Calcola il COM globale pesato per le masse."""
    com_positions = center_of_mass_positions(model, poses)

    total_mass = sum(
        link["mass"]
        for link in model["links"].values()
    )

    if total_mass <= 0:
        raise ValueError("Massa totale non positiva.")

    weighted_sum = np.zeros(3)

    for name, position in com_positions.items():
        weighted_sum += (
            model["links"][name]["mass"] * position
        )

    return weighted_sum / total_mass

def main():
    model = load_urdf("lower_model/lower_body.urdf")

    # Tutti gli angoli non specificati sono zero.
    q = np.array([0.2, 0.1, -0.1, 0.3, -0.2, 0.5, -0.4, 0.2, 0.1, -0.3])
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_dict = {
    "hip_L_flexion": 0.2,
    "hip_L_adduction": 0.1,
    "hip_L_rotation": -0.1,
    "knee_L": 0.3,
    "ankle_L": -0.2,
    "hip_R_flexion": 0.5,
    "hip_R_adduction": -0.4,
    "hip_R_rotation": 0.2,
    "knee_R": 0.1,
    "ankle_R": -0.3
    }
    #q= np.zeros(10)  # Tutti gli angoli a zero

    for k, v in q_dict.items():
        print(k, v)

    poses = forward_kinematics(model, q)
    print("\n=== POSIZIONI DEI LINK ===")

    print("\n=== CENTRO DI MASSA TOTALE ===")
    com = total_center_of_mass(model, poses)
    print("COM globale [m]:", com)

    print("\n=== JACOBIANO DEL PIEDE DESTRO ===")

    # Calcolo del Jacobiano geometrico dal piede destro al mondo
    J = geometric_jacobian(model, q_dict, "foot_R")

    # Confronto del jacobiano con quello calcolato con Pinocchio
    try:
        model_pin = pin.buildModelFromUrdf(str("lower_model/lower_body.urdf"))
    except Exception as exc:
            print("Impossibile costruire il modello Pinocchio.")
            print(exc)
            return None

    q0 = pin.neutral(model_pin)
    # Campire perchè qui parte dalla sinistra per comporre il jacobiano (PER COME LO LEGGE PINOCCHIO SONO SPECULARI MA è AFFIDABILE !!!)
    J_pin = validate_kinematics_pin(model_pin,q)
    print(J-J_pin)

    assert J.shape == (6, 10)
    assert np.all(np.isfinite(J))
    assert np.all(np.isfinite(com))

    print("\n[OK] Test cinematici di base superati.")

# Funzioni Test Pinpcchio
def validate_kinematics_pin(model, q0):
    data = model.createData()


    try:
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)

        frame_id = model.getFrameId("foot_R")

        print("foot_id =", frame_id)
        print("foot frame =", model.frames[frame_id])

        # for joint_id in range(1, model.njoints):
        #     J = pin.computeJointJacobian(
        #         model,
        #         data,
        #         q0,
        #         joint_id,
        #     )

        J = pin.computeFrameJacobian(
            model,
            data,
            q0,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

        # if J.shape == (6, model.nv):
        #     print(
        #         f"Jacobian {model.names[frame_id]:25s} "
        #         f"shape={J.shape}"
        #     )
        # else:
        #     print(
        #         f"Jacobian errato per "
        #         f"{model.names[frame_id]}"
        #     )
        #     return False

    except Exception as exc:
        print(f"Errore nella cinematica: {exc}")
        return False


    return J

if __name__ == "__main__":
    main()

