# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
 Solver D2Q4Q4Q4Q4 for the Euler system

 dt h + dx q_x + dy q_y = 0,
 dt q_x + dx (q_x^2/h + p) + dy (q_xq_y/h) = 0,
 dt q_y + dx (q_xq_y/h) + dy (q_y^2/h + p) = 0,
 dt E + dx ((E+p) q_x/rho) + dy ((E+p) q_y/rho) = 0.
"""
import sympy as sp
import pylbm
import numpy as np

# pylint: disable=redefined-outer-name

gamma = 1.4
mach = 2.5
theta = 15 * np.pi/180


def up2qE(rho, ux, uy, p):
    qx, qy = rho*ux, rho*uy
    E = .5*rho*(ux**2+uy**2) + p/(gamma-1)
    return [rho, qx, qy, E]


def qE2up(rho, qx, qy, E):
    ux, uy = qx/rho, qy/rho
    p = (gamma-1)*(E - .5*rho*(ux**2+uy**2))
    return [rho, ux, uy, p]

def pstar(m, iload, theta):
    # print(iload[0][:, 1:])
    ind = tuple(iload[0][:, 1:].T)
    # import ipdb; ipdb.set_trace()
    rho, ux, uy, p = qE2up(m[RHO][ind], m[QX][ind], m[QY][ind], m[E][ind])

    un = ux * np.sin(theta) - uy * np.cos(theta)
    return p + np.sqrt((p*rho > 0)*gamma*p*rho)*un

def wedge_bc(f, m, m_tot, iload, t, x, y, theta, la):
    ps = pstar(m_tot, iload, theta)[:, np.newaxis]
    ct, st = np.cos(theta), np.sin(theta)
    # f[4] = -ps*st/la
    # f[7] = -ps*ct/la
    # f[8] = ps*ct/la
    # f[11] = ps*st/la
    # f[1], f[5], f[9], f[13] = 0, -ps*ct/la, ps*st/la, 0
    # f[2], f[6], f[10], f[14] = 0, -ps*st/la, ps*ct/la, 0
    m[RHO] = 1
    m[QX] = ps*st
    m[QY] = -ps*ct
    m[E] = .5*ps**2+ps/(gamma-1)

rho_in = 1.
ux_in, uy_in = .7/np.sqrt(3), 0.
v_in = np.sqrt(ux_in**2+uy_in**2)
p_in = (v_in/mach)**2*rho_in/gamma
rho_in, qx_in, qy_in, E_in = up2qE(rho_in, ux_in, uy_in, p_in)
print(f"""
    Initial value
    rho : {rho_in}
    ux  : {ux_in}
    uy  : {uy_in}
    p   : {p_in}
    qx  : {qx_in}
    qy  : {qy_in}
    E   : {E_in}
    c   : {np.sqrt(gamma*p_in/rho_in)}
    """)

X, Y = sp.symbols('X, Y')
RHO, QX, QY, E = sp.symbols('rho, qx, qy, E')
LA, GAMMA = sp.symbols('lambda, gamma', constants=True)
SIGMA_RHOX, SIGMA_RHOXY, SIGMA_QX, SIGMA_QXY, SIGMA_EX, SIGMA_EXY = sp.symbols(
    'sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5', constants=True
)
EpP = GAMMA*E - (GAMMA-1)/2*(QX**2+QY**2)/RHO

# def rho_init(x, y, xmin, xmax, ymin, ymax):
#     """
#     initial condition
#     """
#     center = (
#         .5*xmin + .5*xmax,
#         .5*ymin + .5*ymax
#     )
#     radius = 0.1
#     height = 0.5
#     return 1 + height * ((x-center[0])**2+(y-center[1])**2 < radius**2)


def run(space_step,
        final_time,
        generator="cython",
        sorder=None,
        with_plot=True):
    """
    Parameters
    ----------

    space_step: double
        spatial step

    final_time: double
        final time

    generator: string
        pylbm generator

    sorder: list
        storage order

    with_plot: boolean
        if True plot the solution otherwise just compute the solution


    Returns
    -------

    sol
        <class 'pylbm.simulation.Simulation'>

    """
    # parameters
    xmin, xmax, ymin, ymax = 0., 0.5, 0., 0.5  # bounds of the domain
    xwedge = 0.25*xmax + 0.75*xmin             # wedge position
    la = 4                                     # velocity of the scheme
    sigma = 1/1.65-0.5
    print(1/(.5+sigma))
    sigma_rhox, sigma_rhoxy = sigma, sigma
    sigma_qx, sigma_qxy = sigma, sigma
    sigma_Ex, sigma_Exy = sigma, sigma

    symb_s_rhox = 1/(.5+SIGMA_RHOX)
    symb_s_rhoxy = 1/(.5+SIGMA_RHOXY)
    symb_s_qx = 1/(.5+SIGMA_QX)
    symb_s_qxy = 1/(.5+SIGMA_QXY)
    symb_s_Ex = 1/(.5+SIGMA_EX)
    symb_s_Exy = 1/(.5+SIGMA_EXY)

    s_rho = [0., symb_s_rhox, symb_s_rhox, symb_s_rhoxy]
    s_q = [0., symb_s_qx, symb_s_qx, symb_s_qxy]
    s_E = [0., symb_s_Ex, symb_s_Ex, symb_s_Exy]

    vitesse = list(range(1, 5))
    polynomes = [1, X, Y, X**2-Y**2]

    simu_cfg = {
        'parameters': {
            LA: la,
            GAMMA: gamma,
            SIGMA_RHOX: sigma_rhox,
            SIGMA_RHOXY: sigma_rhoxy,
            SIGMA_QX: sigma_qx,
            SIGMA_QXY: sigma_qxy,
            SIGMA_EX: sigma_Ex,
            SIGMA_EXY: sigma_Exy
        },
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [0, 0, 1, 1]
        },
        'elements': [
            pylbm.Triangle(
                (xwedge, 0),
                (xmax, 0),
                (xmax, xmax*np.tan(theta)),
                label=2
            ),
        ],
        'space_step': space_step,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': vitesse,
                'conserved_moments': RHO,
                'polynomials': polynomes,
                'relaxation_parameters': s_rho,
                'equilibrium': [RHO, QX, QY, 0.],
            },
            {
                'velocities': vitesse,
                'conserved_moments': QX,
                'polynomials': polynomes,
                'relaxation_parameters': s_q,
                'equilibrium': [
                    QX,
                    (GAMMA-1) * E + .5*(
                        (3-GAMMA)*QX**2+(1-GAMMA)*QY**2
                    ) / RHO,
                    QX*QY/RHO,
                    0.
                ],
            },
            {
                'velocities': vitesse,
                'conserved_moments': QY,
                'polynomials': polynomes,
                'relaxation_parameters': s_q,
                'equilibrium': [
                    QY,
                    QX*QY/RHO,
                    (GAMMA-1) * E + .5*(
                        (3-GAMMA)*QY**2+(1-GAMMA)*QX**2
                    ) / RHO,
                    0.
                ],
            },
            {
                'velocities': vitesse,
                'conserved_moments': E,
                'polynomials': polynomes,
                'relaxation_parameters': s_E,
                'equilibrium': [
                    E,
                    EpP * QX / RHO,
                    EpP * QY / RHO,
                    0.
                ],
            },
        ],
        'init': {
            RHO: rho_in,
            QX: qx_in,
            QY: qy_in,
            E: E_in
        },
        'boundary_conditions': {
            0: {
                'method': {
                    0: pylbm.bc.NeumannX,
                    1: pylbm.bc.NeumannX,
                    2: pylbm.bc.NeumannX,
                    3: pylbm.bc.NeumannX
                },
            },
            1: {
                'method': {
                    0: pylbm.bc.NeumannY,
                    1: pylbm.bc.NeumannY,
                    2: pylbm.bc.NeumannY,
                    3: pylbm.bc.NeumannY
                },
            },
            2: {
                'method': {
                    0: pylbm.bc.BouzidiBounceBack,
                    1: pylbm.bc.BouzidiBounceBack,
                    2: pylbm.bc.BouzidiBounceBack,
                    3: pylbm.bc.BouzidiBounceBack
                },
                'value': (wedge_bc, (theta, la)),
                'time_bc': True,
            },
        },
        'generator': generator,
        'show_code': True,
        # 'codegen_option': {'directory': 'wedge_codegen'}
        }

    # build the simulations
    sol = pylbm.Simulation(simu_cfg, sorder=sorder)

    if with_plot:
        # create the viewer to plot the solution
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        axe = fig[0]

        p = (gamma-1)*(sol.m[E] - .5*(sol.m[QX]**2+sol.m[QY]**2)/sol.m[RHO])

        axe.plot(
            [xwedge, xmax],
            [0, (xmax-xwedge)*np.tan(theta)],
            width=2, color='black', alpha=1
        )
        axe.plot(
            [xwedge, xmax],
            [0, (xmax-xwedge)*np.tan(theta)],
            width=2, color='black', alpha=1
        )
        surf = axe.SurfaceImage(p, cmap='jet')

        def update(iframe):  # pylint: disable=unused-argument
            if sol.t < final_time:
                for _ in range(100):
                    sol.one_time_step()
                p = (gamma-1)*(sol.m[E] - .5*(sol.m[QX]**2+sol.m[QY]**2)/sol.m[RHO])
                vmin, vmax = np.amin(p), np.amax(p)
                print(vmin, vmax)
                surf.img.set_clim(vmin=vmin, vmax=vmax)
                surf.update(p)

        fig.animate(update, interval=1)
        fig.show()
    else:
        while sol.t < final_time:
            sol.one_time_step()

    return sol


if __name__ == '__main__':
    # pylint: disable=invalid-name
    space_step = 1./128
    final_time = 20
    run(space_step, final_time, generator='numpy')
