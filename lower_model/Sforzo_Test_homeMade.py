# ============================================================
# CONFIGURAZIONE
# ============================================================
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from urdf_parser import load_urdf
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time

from TestKinNoPin import(
    forward_kinematics,
    geometric_jacobian,    
)

from TestDynNoPin import (
    mass_matrix,
    gravity_vector,
    coriolis_vector,
    inverse_dynamics,
    q_dict_from_vector
)

# Path Dell'URDF
URDF_PATH = "lower_model/lower_body.urdf"

T_EXT = 0.5       # Durata estensione [s]
T_PAUSE = 0.2     # Pausa a ginocchio esteso [s]
T_FLEX = 1.0      # Durata ritorno [s]
DT = 0.05         # Passo temporale [s]

T_TOTAL = T_EXT + T_PAUSE + T_FLEX

N_DOF = 10

# q0 = np.array([np.pi/2, 0.0, 0.0, -np.pi/2, np.pi/2, np.pi/2, 0.0, 0.0, -np.pi/2, np.pi/2])

# Indici delle coordinate nel vettore q (base zero).
# Adatta questi indici all'ordine effettivo del tuo modello.
# IDX_PELVIS = 0
IDX_HIP_FLEXION_L = 0
IDX_KNEE_L = 3
IDX_ANKLE_L = 4
IDX_HIP_FLEXION_R = 5
IDX_KNEE_R = 8
IDX_ANKLE_R = 9


# Angoli iniziali e finali del ginocchio [rad]
KNEE_START = np.deg2rad(-90.0)
KNEE_END = np.deg2rad(10.0)

# Angoli da mantenere fissi idealmentee
HIP_FLEXION_R_FIXED = np.deg2rad(90.0)
ANKLE_R_FIXED = np.deg2rad(90.0)
HIP_FLEXION_L_FIXED = np.deg2rad(90.0)
KNEE_L_FIXED = np.deg2rad(-90.0)
ANKLE_L_FIXED = np.deg2rad(90.0)


# ============================================================
# PROFILO QUINTICO
# ============================================================

def quintic(u):
    """Posizione normalizzata con velocità e accelerazione nulle
    agli estremi. u deve appartenere a [0, 1].
    """
    u = np.clip(u, 0.0, 1.0)
    # Equazione temporale del 5 ordine e le sue derivate
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = 30*u**2 - 60*u**3 + 30*u**4
    dds = 60*u - 180*u**2 + 120*u**3
    return s, ds, dds


def knee_trajectory(t):
    """
    Restituisce angolo, velocità e accelerazione del ginocchio.
    Unità: rad, rad/s, rad/s^2.
    """
    delta = KNEE_END - KNEE_START

    if t <= T_EXT:
        u = t / T_EXT
        s, ds, dds = quintic(u)

        q = KNEE_START + delta * s
        qd = delta * ds / T_EXT
        qdd = delta * dds / T_EXT**2

    elif t <= T_EXT + T_PAUSE:
        q = KNEE_END
        qd = 0.0
        qdd = 0.0

    else:
        tau = t - T_EXT - T_PAUSE
        u = tau / T_FLEX
        s, ds, dds = quintic(u)

        q = KNEE_END + (KNEE_START - KNEE_END) * s
        qd = (KNEE_START - KNEE_END) * ds / T_FLEX
        qdd = (KNEE_START - KNEE_END) * dds / T_FLEX**2

    return q, qd, qdd


# ============================================================
# GENERAZIONE DELLA TRAIETTORIA ARTICOLARE
# ============================================================

def generate_trajectory():
    time = np.arange(0.0, T_TOTAL + DT/2, DT)

    q_all = np.zeros((len(time), N_DOF))
    qd_all = np.zeros_like(q_all)
    qdd_all = np.zeros_like(q_all)

    for i, t in enumerate(time):
        # # Bacino fisso
        # q_all[i, IDX_PELVIS] = 0.0

        # Settaggio per i Joint che devono stare fissi (logica da migliorare)
        q_all[i, IDX_HIP_FLEXION_R] = HIP_FLEXION_R_FIXED
        q_all[i, IDX_HIP_FLEXION_L] = HIP_FLEXION_L_FIXED
        q_all[i, IDX_KNEE_L] = KNEE_L_FIXED
        q_all[i, IDX_ANKLE_R] = ANKLE_R_FIXED
        q_all[i, IDX_ANKLE_L] = ANKLE_L_FIXED


        # Ginocchio destro: unico giunto attivo
        q, qd, qdd = knee_trajectory(t)

        q_all[i, IDX_KNEE_R] = q
        qd_all[i, IDX_KNEE_R] = qd
        qdd_all[i, IDX_KNEE_R] = qdd

    return time, q_all, qd_all, qdd_all


# ============================================================
# ADATTATORI: COLLEGA QUI LE TUE FUNZIONI
# ============================================================

def compute_fk(q):
    """
    Sostituisci con la tua funzione di cinematica diretta.
    Esempio:
        return tua_cinematica_diretta(q)
    """
    raise NotImplementedError("Collega la funzione FK esistente")


def compute_dynamics(q, qd, qdd):
    """
    Sostituisci con la tua funzione di dinamica inversa.
    Restituisci, ad esempio, il vettore delle coppie articolari.
    """
    raise NotImplementedError("Collega la funzione di dinamica esistente")


# ============================================================
# ESECUZIONE E SALVATAGGIO
# ============================================================

def run_simulation(model):
    time, q_all, qd_all, qdd_all = generate_trajectory()
    tau_all = np.zeros((len(time), N_DOF))

    results_trajectory = []

    for i, t in enumerate(time):
        q = q_all[i]
        qd = qd_all[i]

        # Verificare quanto rumorosa
        qdd = qdd_all[i]

        q_dict = q_dict_from_vector(q)

        # Calcolo postura rispetto T0i mondo-link i esimo (Appesantisce troppo la sovrascittura in piu)
        # poses = forward_kinematics(model, q_dict) 


        # Calcolo della coppia generata durante un certo deltaT (Tanto anche qui si appoggia alle pose)
        tau = inverse_dynamics(model, q, qd, qdd)
        poses = None

        # Con pinocchio        
        # data = model.createData()
        # pin.forwardKinematics(model, data, q)
        # poses = pin.updateFramePlacements(model, data)
        
        # tau = pin.rnea(
        #     model,
        #     data,
        #     q,
        #     qd,
        #     qdd
        # )
        tau_all[i] = tau

        results_trajectory.append({
            "time": t,
            "q": q.copy(),
            "qd": qd.copy(),
            "qdd": qdd.copy(),
            "pose": poses,
            "tau": tau,
        })

    # CSV dei segnali articolari
    with open("leg_extension_trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)

        header = ["time_s"]
        header += [f"q{i+1}_rad" for i in range(N_DOF)]
        header += [f"qd{i+1}_rad_s" for i in range(N_DOF)]
        header += [f"qdd{i+1}_rad_s2" for i in range(N_DOF)]
        header += [f"tau{i+1}" for i in range(N_DOF)]

        writer.writerow(header)

        for i, t in enumerate(time):
            writer.writerow(
                [t]
                + q_all[i].tolist()
                + qd_all[i].tolist()
                + qdd_all[i].tolist()
                + tau_all[i].tolist()
            )

    return results_trajectory

def main():

    # Leggo modello
    if(0):
        model = load_urdf(URDF_PATH)
        model_pin = pin.buildModelFromUrdf(URDF_PATH)
        # Genero traiettoia (per adesso solo leg extention)
        data_traj = run_simulation(model)


        #### PARTE SIMULAZIONE 3D    
        # time_array = np.array([item["time"] for item in data_traj])
        # q_traj = np.array([item["q"] for item in data_traj])

        # model, collision_model, visual_model = pin.buildModelsFromUrdf(
        #     URDF_PATH,

        # )

        # viz = MeshcatVisualizer(
        #     model,
        #     collision_model,
        #     visual_model
        # )

        # viz.initViewer(open=False)
        # print("\nMeshcat:")
        # print(viz.viewer.url())

        # # Carica il modello
        # viz.loadViewerModel()

        # for t, q in zip(time_array, q_traj):
        #     viz.display(q)
        #     time.sleep(10*DT)

        # print("Animazione terminata.")

    else:
        CSV_PATH = "C:/Users/Tommas0/Desktop/Kinematic/KineLab2026/leg_extension_trajectory.csv"

        df = pd.read_csv(CSV_PATH)
        t = df["time_s"]

        # Seleziona l'indice del ginocchio destro nel vettore q.
        # Nel CSV precedente il ginocchio è già salvato anche
        # esplicitamente in coordinate angolari.
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # 1. Angolo del ginocchio
        axes[0].plot(t, df["q9_rad"], label="Ginocchio destro")
        axes[0].set_ylabel("Angolo [deg]")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Velocità angolare
        axes[1].plot(
            t, df["qd9_rad_s"],
            label="Velocità ginocchio"
        )
        axes[1].set_ylabel("Velocità [rad/s]")
        axes[1].legend()
        axes[1].grid(True)

        # 3. Accelerazione angolare
        axes[2].plot(
            t, df["qdd9_rad_s2"],
            label="Accelerazione ginocchio"
        )
        axes[2].set_ylabel("Accelerazione [rad/s²]")
        axes[2].legend()
        axes[2].grid(True)

        # 4. Coppie articolari
        tau_cols = [c for c in df.columns if c.startswith("tau")]

        for col in tau_cols:
            axes[3].plot(t, df[col], label=col)

        axes[3].set_ylabel("Coppia [Nm]")
        axes[3].set_xlabel("Tempo [s]")
        axes[3].legend(ncol=2)
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig("leg_extension_results.png", dpi=200)
        plt.show()


    
if __name__ == "__main__":
    main()
    # time, q, qd, qdd = generate_trajectory()
    # print(f"Durata: {T_TOTAL:.2f} s")
    # print(f"Campioni: {len(time)}")
    # print(f"Forma q: {q.shape}")
    # print(f"Angolo iniziale ginocchio: {np.rad2deg(q[0, IDX_KNEE_R]):.1f} deg")
    # print(f"Angolo finale ginocchio: {np.rad2deg(q[-1, IDX_KNEE_R]):.1f} deg")
