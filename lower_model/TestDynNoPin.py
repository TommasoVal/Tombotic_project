import numpy as np
import pinocchio as pin
from urdf_parser import load_urdf

from TestKinNoPin import (
    total_center_of_mass,
    forward_kinematics,
    geometric_jacobian,
    rotation_rpy,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

GRAVITY_WORLD = np.array([
    0.0,
    0.0,
    -9.81,
])


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

LOAD_PROPRIETIES = {
    "mass": 10,
    "solidal_link": "shank_R",
    "local_position": np.array([0.0, 0.0, -0.43000]),
}


# ============================================================
# UTILITY
# ============================================================

def get_movable_joints(model):
    """
    Restituisce i giunti mobili nell'ordine utilizzato
    da tutto il modulo dinamico.
    """

    return [
        model["joints"][name]
        for name in JOINT_ORDER
    ]


def q_dict_from_vector(q):
    """
    Converte un vettore q nell'ordine JOINT_ORDER
    in un dizionario {joint_name: value}.
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (len(JOINT_ORDER),):
        raise ValueError(
            f"q deve avere shape "
            f"({len(JOINT_ORDER)},), ricevuta {q.shape}"
        )

    return {
        name: q[i]
        for i, name in enumerate(JOINT_ORDER)
    }


# ============================================================
# CINEMATICA TEMPORALE
# ============================================================

def joint_velocities(q_prev, q_next, dt):
    """
    Calcola q_dot tramite differenza centrale/semplice
    tra due configurazioni.

    q_prev : configurazione precedente
    q_next : configurazione successiva
    dt     : intervallo temporale [s]
    """

    q_prev = np.asarray(q_prev, dtype=float)
    q_next = np.asarray(q_next, dtype=float)

    if dt <= 0:
        raise ValueError("dt deve essere > 0")

    if q_prev.shape != q_next.shape:
        raise ValueError("q_prev e q_next devono avere la stessa shape")

    return (q_next - q_prev) / dt

# RUMOROSO PROBABILMENTE NON USATO, MA PUO' ESSERE UTILE PER TEST
def joint_accelerations(q_prev, q, q_next, dt):
    """
    Calcola q_ddot tramite differenza centrale: (fare attenzione se dt non è costante)

        q_ddot = (q_next - 2q + q_prev) / dt^2
    """

    q_prev = np.asarray(q_prev, dtype=float)
    q = np.asarray(q, dtype=float)
    q_next = np.asarray(q_next, dtype=float)

    if dt <= 0:
        raise ValueError("dt deve essere > 0")

    return (
        q_next
        - 2.0 * q
        + q_prev
    ) / (dt ** 2)


def trajectory_derivatives(q_trajectory, dt):
    """
    Calcola velocità e accelerazioni di una traiettoria.

    Parametri
    ---------
    q_trajectory : array (N, n)
        Traiettoria articolare.
    dt : float
        Periodo di campionamento.

    Returns
    -------
    q_dot : array (N, n)
    q_ddot : array (N, n)

    np.gradient viene usato lungo l'asse temporale.
    """

    q_trajectory = np.asarray(q_trajectory, dtype=float)

    if q_trajectory.ndim != 2:
        raise ValueError(
            "q_trajectory deve avere shape (N, n)"
        )

    q_dot = np.gradient(
        q_trajectory,
        dt,
        axis=0,
        edge_order=2,
    )

    q_ddot = np.gradient(
        q_dot,
        dt,
        axis=0,
        edge_order=2,
    )

    return q_dot, q_ddot


# ============================================================
# MATRICE DI MASSA
# ============================================================

def mass_matrix(model, q, link_load=None, point_load=None, mass_load=None):

    #### AGGIUNGO POSSIBILITA DI PRESENZA DI UN CARICO PUNTIFORME SU UN PUNTO DEL CORPO SOLIDALE AD UN LINK:
        ## point_load indica le coordinate locali rispetto link di riferimento e link_load il link solidale al peso
        ## Per Test Lunghezza_Tibia = 0.430000 = Lunghezza_coscia      

    """
    Calcola la matrice di massa M(q).

    M è ottenuta tramite la formulazione energetica:

        M = sum[
            m Jv^T Jv
            +
            Jw^T I Jw
        ]

    """
    q_dict = q_dict_from_vector(q)

    poses = forward_kinematics(
        model,
        q_dict,
    )

    movable_joints = get_movable_joints(model)
    n = len(movable_joints)

    M = np.zeros((n, n))

    for link_name, link in model["links"].items():

        mass = link["mass"]

        if mass <= 0:
            continue

        if not link["has_inertial"]:
            continue

        # Jacobiano del COM del link
        J = geometric_jacobian(
            model,
            q_dict,
            link_name,
            point_local=link["com"],
        )

        Jv = J[:3, :]
        Jw = J[3:, :]

        # Orientamento del link
        R_world_link = poses[link_name][:3, :3]

        # Orientamento del frame inerziale
        R_link_inertial = rotation_rpy(
            link["inertial_origin"]["rpy"]
        )

        R_world_inertial = (
            R_world_link @ R_link_inertial
        )

        # Inerzia definita nel frame inerziale
        I_inertial = link["inertia"]

        # Inerzia espressa nel mondo
        I_world = (
            R_world_inertial
            @ I_inertial
            @ R_world_inertial.T
        )

        # Contributo traslazionale
        M += mass * (Jv.T @ Jv)

        # Contributo rotazionale
        M += Jw.T @ I_world @ Jw

    # Calcolo aggiuntivo del Jacobiano qui 
    if point_load is None and mass_load is None:
        print("Non è definito alcun carico")
    else:
        if link_load is None:
            print("Non definito in che parte del corpo è applicato carico")
        else:
            J_l = geometric_jacobian(
                model,
                q_dict,
                link_load,
                point_local=point_load,
            )
            J_lv = J_l[:3,:]
            J_lw= J_l[3:,:] # Inutile se non ha inerzia (Nel caso si definisse bisogna aggiungere termine rotazionale)
            M += mass_load * (J_lv.T @ J_lv)

    # Eliminiamo eventuali piccoli errori numerici
    M = 0.5 * (M + M.T)

    return M


# ============================================================
# TERMINE GRAVITAZIONALE
# ============================================================

def gravity_vector(model, q, link_load=None, point_load=None, mass_load=None, link_v_name=None):
    
    #### AGGIUNGO VINCOLO SEDILE SU COSCIA (DA CAPIRE SE VA BENE CONSIDERARE SOLO MASSA COSCIA):
    
    """
    Calcola g(q), cioè il vettore dei momenti generalizzati
    necessari per compensare la gravità.
    """
    
    q_dict = q_dict_from_vector(q)

    n = len(JOINT_ORDER)

    g = np.zeros(n)

    for link_name, link in model["links"].items():

        mass = link["mass"]

        if mass <= 0:
            continue

        if not link["has_inertial"]:
            continue

        J = geometric_jacobian(
            model,
            q_dict,
            link_name,
            point_local=link["com"],
        )

        Jv = J[:3, :]

        g -= Jv.T @ (
            mass * GRAVITY_WORLD
        )

    # Calcolo aggiuntivo del Jacobiano qui 
    if point_load is None and mass_load is None:
        print("Non è definito alcun carico")
    else:
        if link_load is None:
            print("Non definito in che parte del corpo è applicato carico")
        else:
            J_l = geometric_jacobian(
                model,
                q_dict,
                link_load,
                point_local=point_load,
            )    
            J_lv = J_l[:3,:]

            g -= J_lv.T @ (
            mass_load * GRAVITY_WORLD
            )

    # Contributo reazione vincolare    
    if link_v_name is None:
        print("Nessun vincolo aggiunto")
    else:
        for link_v in link_v_name:
            if link_v not in model["links"]:
                print(f"Link sconosciuto: {link_v}")
            else:   
                # DOPO VERIFICA ACCORPA SEMPLIFICANDO DALLA PRIMA NELLE OPERAZIONI, QUESTO E SOLO DI TEST PER CALCOLI
                mass_v = model["links"][link_v]["mass"]
                
                if mass_v <= 0:
                    continue
        
                if not model["links"][link_v]["has_inertial"]:
                    continue

                J_v = geometric_jacobian(
                    model,
                    q_dict,
                    link_v,
                    point_local=model["links"][link_v]["com"],
                )    
                J_vv = J_v[:3,:]

                g += J_vv.T @ (
                mass_v * GRAVITY_WORLD
                )
    
    return g


# ============================================================
# MATRICE DI CORIOLIS / CENTRIFUGA
# ============================================================

def mass_matrix_derivatives(model, q, epsilon=1e-6):
    """
    Calcola numericamente le derivate della matrice M(q):

        dM/dq_k

    Restituisce un array:

        dM[k, i, j]

    """

    q = np.asarray(q, dtype=float)

    n = len(q)

    dM = np.zeros((n, n, n))

    for k in range(n):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[k] += epsilon
        q_minus[k] -= epsilon

        M_plus = mass_matrix(
            model,
            q_plus,
            LOAD_PROPRIETIES["solidal_link"],
            LOAD_PROPRIETIES["local_position"],
            LOAD_PROPRIETIES["mass"],
        )

        M_minus = mass_matrix(
            model,
            q_minus,
            LOAD_PROPRIETIES["solidal_link"],
            LOAD_PROPRIETIES["local_position"],
            LOAD_PROPRIETIES["mass"],
        )

        dM[k] = (
            M_plus - M_minus
        ) / (2.0 * epsilon)

    return dM


def coriolis_vector(model, q, q_dot, epsilon=1e-6):
    """
    Calcola direttamente il termine:

        C(q, q_dot) q_dot

    utilizzando i simboli di Christoffel.

    Formula:

        c_i =
            sum_j sum_k Gamma_ijk q_dot_j q_dot_k

    con:

        Gamma_ijk =
            1/2 [
                dM_ij/dq_k
                +
                dM_ik/dq_j
                -
                dM_jk/dq_i
            ]
    """

    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)

    n = len(q)

    if q_dot.shape != q.shape:
        raise ValueError(
            "q e q_dot devono avere la stessa shape"
        )

    dM = mass_matrix_derivatives(
        model,
        q,
        epsilon=epsilon,
    )

    c = np.zeros(n)

    for i in range(n):

        for j in range(n):

            for k in range(n):

                Gamma = 0.5 * (
                    dM[k, i, j]
                    + dM[j, i, k]
                    - dM[i, j, k]
                )

                c[i] += (
                    Gamma
                    * q_dot[j]
                    * q_dot[k]
                )

    return c


# ============================================================
# DINAMICA INVERSA
# ============================================================

def inverse_dynamics(
    model,
    q,
    q_dot,
    q_ddot,
):
    """
    Dinamica inversa:

        tau =
            M(q) q_ddot
            +
            C(q,q_dot) q_dot
            +
            g(q)

    """

    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)
    q_ddot = np.asarray(q_ddot, dtype=float)

    n = len(JOINT_ORDER)

    if q.shape != (n,):
        raise ValueError("q shape non valida")

    if q_dot.shape != (n,):
        raise ValueError("q_dot shape non valida")

    if q_ddot.shape != (n,):
        raise ValueError("q_ddot shape non valida")

    M = mass_matrix(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
    )

    g = gravity_vector(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
        ["thigh_L","thigh_R"],
    )

    c = coriolis_vector(
        model,
        q,
        q_dot,
    )

    tau_inertia = M @ q_ddot

    tau = (
        tau_inertia
        + c
        + g
    )

    return tau


# ============================================================
# OUTPUT COMPLETO
# ============================================================

def dynamics_terms(
    model,
    q,
    q_dot,
    q_ddot,
):
    """
    Restituisce separatamente tutti i termini dinamici.
    """

    M = mass_matrix(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
    )

    g = gravity_vector(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
        ["thigh_L","thigh_R"],
    )

    c = coriolis_vector(
        model,
        q,
        q_dot,
    )

    inertia = M @ q_ddot

    tau = (
        inertia
        + c
        + g
    )

    return {
        "M": M,
        "gravity": g,
        "coriolis": c,
        "inertia": inertia,
        "tau": tau,
    }
def main():
    """
    Test di esempio.
    """

    model = load_urdf("lower_model/lower_body.urdf")

    # q = np.zeros(len(JOINT_ORDER))
    # q_dot = np.zeros(len(JOINT_ORDER))

    q = np.array([0.2, 0.1, -0.1, 0.3, -0.2, 0.5, -0.4, 0.2, 0.1, -0.3])
    q_dot = np.array([0.0, 0.0, -0.0, 0.1, -0.1, 0.0, -0.0, 0.0, 0.1, -0.1])
    q_ddot = np.zeros(len(JOINT_ORDER))

    terms = dynamics_terms(
        model,
        q,
        q_dot,
        q_ddot,
    )

    g_1 = gravity_vector(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
    )
    g_2 = gravity_vector(
        model,
        q,
        LOAD_PROPRIETIES["solidal_link"],
        LOAD_PROPRIETIES["local_position"],
        LOAD_PROPRIETIES["mass"],
        ["thigh_L","thigh_R"],
    )

    print(g_1)
    print(g_2)

    poses = forward_kinematics(model, q_dict_from_vector(q))
    com = total_center_of_mass(model, poses)

    ## Parte pinocchio
    model_pin = pin.buildModelFromUrdf(str("lower_model/lower_body.urdf"))
    # Se il modello Pinocchio è a base fissa:
    assert model_pin.nq == len(JOINT_ORDER)
    assert model_pin.nv == len(JOINT_ORDER)

    data = model_pin.createData()
    M_pin = pin.crba(model_pin, data, q)
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T

    com_pin = pin.centerOfMass(model_pin, data, q)
    # Per non escludere pelvis(pinocchio calcola il COM escludendo l'universe link)
    total_mass_pin = pin.computeTotalMass(model_pin)
    com_pin = (total_mass_pin * com_pin + model_pin.inertias[0].mass * model_pin.inertias[0].lever) / (model_pin.inertias[0].mass + total_mass_pin)
    C_pin = pin.computeCoriolisMatrix(model_pin, data, q, q_dot)
    g_pin = pin.computeGeneralizedGravity(model_pin, data, q)

    tau_pin = pin.rnea(
        model_pin,
        data,
        q,
        q_dot,
        q_ddot
    )

    # for jid, name in enumerate(model_pin.names):
    #     inertia = model_pin.inertias[jid]
    #     print(
    #         name,
    #         "massa =", inertia.mass,
    #         "COM locale =", inertia.lever
    #     )

    print("center of mass:")
    print(com - com_pin)
    print("Massa totale:", pin.computeTotalMass(model_pin))
    print("M(q):")
    print(terms["M"]-M_pin)
    print()
    print("g(q):")
    print(terms["gravity"]-g_pin)
    print()
    print("C(q,q_dot)q_dot:")
    print(terms["coriolis"]-C_pin@q_dot)
    print()
    print("M(q)q_ddot:")
    print(terms["inertia"])
    print()
    print("tau:")
    print(terms["tau"])
    print("tau (Pinocchio):")
    print(tau_pin-terms["coriolis"]-terms["gravity"])

if __name__ == "__main__":
    main()