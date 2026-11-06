import sys
import argparse
from pathlib import Path
import numpy as np

from urdf_parser import load_urdf
from TestKinNoPin import (
    total_center_of_mass,
    forward_kinematics,
    geometric_jacobian,
    rotation_rpy,
)
from TestDynNoPin import (
    q_dict_from_vector,
    get_movable_joints
)

# Aggiungi il percorso della cartella che contiene la libreria
add_directory = 'C:/Users/Tommas0/Desktop/Kinematic/KineLab2026/TestEnvironment/tools'
sys.path.insert(0, add_directory)
ROOT = Path(__file__).resolve().parents[1]

from plot_kinematic_json import load_data


def trajectory_from_json(json_file):
    try:
        data, devices = load_data(json_file)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"Errore lettura JSON: {exc}")
        return None

    # Estrai i dati di traiettoria dal JSON
    trajectory_data = []
    for i, device in enumerate(devices):
        samples = device["samples"]
        name = device["configuration"]["name"]
        # trajectory_data.append(samples)
        trajectory_data.append({name: samples})  # Salva i campioni con il nome del dispositivo come chiave

    # Logica di ottenimento posa articolari    

    return trajectory_data

def rotation_x(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=float)


def rotation_y(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=float)


def rotation_z(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=float)

def hip_rotation_from_angles(q_hip):
    """
    Rotazione bacino -> coscia per la catena:
    Rx(q_flexion) @ Ry(q_adduction) @ Rz(q_rotation).

    Valido se gli origin rpy dei tre giunti sono nulli.
    """
    q1, q2, q3 = np.asarray(q_hip, dtype=float)

    return (
        rotation_x(q1)
        @ rotation_y(q2)
        @ rotation_z(q3)
    )


def hip_angles_from_rotation(R):
    """
    Ricava gli angoli dell'anca da una rotazione relativa.

    Ordine restituito:
        [hip_flexion, hip_adduction, hip_rotation]

    Angoli in radianti.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("R deve essere una matrice 3x3")

    # Protegge asin da piccoli errori numerici.
    value = np.clip(R[0, 2], -1.0, 1.0)

    q1 = np.arctan2(-R[1, 2], R[2, 2])
    q2 = np.arcsin(value)
    q3 = np.arctan2(-R[0, 1], R[0, 0])

    return np.array([q1, q2, q3])


def main():
    # parte di parsing degli argomenti per evitare conflitti con l'importazione
    parser = argparse.ArgumentParser(description="Plotta gyro e accelerazione del JSON Cinematica.")
    parser.add_argument("file", nargs="?", type=Path, default= ROOT / "TestEnvironment/cinematica_simulata.json")
    # parser.add_argument("--body-part", type=int, help="Filtra per body_part_id (4 = gamba destra).")
    parser.add_argument("--check", action="store_true", help="Verifica il file senza aprire finestre.")
    args = parser.parse_args()

    # q di test
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  
    q_dict = q_dict_from_vector(q)
    
    model = load_urdf("model/lower_body.urdf")
    trajectory_data = trajectory_from_json(args.file)

    # for i, samples in enumerate(trajectory_data[]):

    poses = forward_kinematics(model, q_dict)
    print("Posizioni dei segmenti:")
    for side in ("R", "L"):
        R_test = poses[f"thigh_{side}"][:3, :3]  # Estrai la matrice di rotazione dalla posa dell'anca {side}
        qx, qy, qz = hip_angles_from_rotation(R_test)  # Calcola la rotazione dell'anca dai primi tre angoli 
        print(f"Angoli dell'anca {side} calcolati: qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}")
    #Cinematica inversa

    # for device in devices:
    #     config = device["configuration"]
    #     print(f"Device {config['id']} / parte {config['body_part_id']}: "
    #          f"{len(device['samples'])} campioni")
    #     print(data)
    

if __name__ == "__main__":
    main()
  