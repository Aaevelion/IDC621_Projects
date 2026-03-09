import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kuramoto Model Simulation with All-to-All Coupling

    The Kuramoto model describes a population of *N* oscillators, each with its own natural frequency $\omega_i$, that are weakly coupled together, and tries to model the eventual spontaneous synchronisation of these coupled oscillators. The governing equation for each oscillator's phase $\theta_i$ (assuming all-to-all coupling) is:

    $$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)$$

    The first term makes each oscillator spin at its own rate. The second term is the coupling: every oscillator is nudged toward all the others, with strength *K*. The factor 1/*N* keeps the total coupling finite as the population grows.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec
    import os
    import marimo as mo

    file = mo.notebook_location()
    return Circle, Line2D, animation, file, mo, np, os, plt


@app.cell
def _(np):
    # ─────────────────────────────────────────────────────────────
    #  GLOBAL PARAMETERS
    # ─────────────────────────────────────────────────────────────
    RNG_SEED   = 42          # reproducibility
    N          = 60          # number of oscillators
    DT         = 0.05        # Euler integration time-step  (seconds)
    T_TOTAL    = 25.0        # total simulation time        (seconds)
    GAMMA      = 0.5         # half-width of Lorentzian g(ω); K_c = 2γ = 1.0
    N_STEPS    = int(T_TOTAL / DT)

    # Critical coupling for a Lorentzian distribution: K_c = 2γ
    K_CRITICAL = 2.0 * GAMMA   # = 1.0

    # Coupling strengths used in the multi-K animation
    K_VALUES   = [0.3, 0.8, 1.0, 1.5, 2.5]
    K_COLORS   = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

    # Coupling strengths for the bifurcation-sweep animation
    K_SWEEP    = np.linspace(0.0, 3.0, 40)
    return (
        DT,
        GAMMA,
        K_COLORS,
        K_CRITICAL,
        K_SWEEP,
        K_VALUES,
        N,
        N_STEPS,
        RNG_SEED,
        T_TOTAL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Natural Frequencies and Quenched Disorder

    Each oscillator is assigned a natural frequency $\omega_i$ drawn from a Lorentzian (Cauchy) distribution:

    $$g(\omega) = \frac{\gamma/\pi}{\gamma^2 + \omega^2}$$

    These frequencies are fixed at the start and never change — this is called *quenched disorder*. The Lorentzian is chosen because it gives an exact analytical solution, which lets us check the simulation directly against theory. We set $\gamma$ = 0.5, which determines the spread of frequencies in the population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Measuring Synchronisation: The Order Parameter

    To quantify how synchronised the population is at any moment, we use the complex order parameter:

    $$r \, e^{i\psi} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}$$

    Geometrically, this is just the centroid of all the oscillators plotted on the unit circle. When phases are spread randomly around the circle, $r\sim 0$ (incoherent). When they all cluster together, $r\sim 1$ (fully synchronised).
    """)
    return


@app.cell
def _(
    DT,
    K_CRITICAL,
    K_SWEEP,
    K_th,
    arrow_op,
    bif_r,
    centroid_dot,
    current_vline,
    frames2,
    frames_circ,
    lines2,
    np,
    osc_colors,
    psi_hist_d,
    r_hist_d,
    r_line,
    r_text_c,
    r_th,
    scat_pts,
    sim_scat,
    status_txt,
    th_hist,
    th_line,
    time_axis,
    time_text_c,
    time_txt2,
    trajs,
):
    # ─────────────────────────────────────────────────────────────
    #  NATURAL FREQUENCY SAMPLING
    def sample_lorentzian(N, gamma, rng):
        """
        Draw N natural frequencies from the Lorentzian (Cauchy) distribution:

            g(ω) = γ/π / (γ² + ω²)

        This is the distribution used in Kuramoto's original exact solution.
        The critical coupling is K_c = 2γ.  We use numpy's standard Cauchy
        sampler scaled by γ and centred at zero so the mean is zero,
        which corresponds to working in the rotating frame of Ω = 0.
        """
        return gamma * rng.standard_cauchy(N)

    def order_parameter(theta):
        """
        Compute the complex order parameter r·e^{iψ} (Eq. 2 of paper).
    #  KURAMOTO DYNAMICS  (vectorised Euler integrator)

        Parameters
        ----------
        theta : (N,) array of oscillator phases [radians]

        Returns
        -------
        r   : float – synchronisation amplitude  ∈ [0, 1]
        psi : float – mean phase [radians]
        """
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        return (r, psi)

    def kuramoto_mfa(theta, omega, K, N):
        """
        Right-hand side of the Kuramoto equation for all-to-all coupling (Eq. 3):

            dθ_i/dt = ω_i + K·r·sin(ψ - θ_i)

        Using the mean-field form is O(N) instead of the naive O(N²) double sum.
        Both formulations are mathematically equivalent for global coupling.

        Parameters
        ----------
        theta : (N,) current phases
        omega : (N,) natural frequencies
        K     : float coupling strength
        N     : int  number of oscillators

        Returns
        -------
        dtheta_dt : (N,) time derivatives of phases
        r         : float  order parameter amplitude
        psi       : float  order parameter phase
        """
        r, psi = order_parameter(theta)
        dtheta_dt = omega + _K * r * np.sin(psi - theta)
        return (dtheta_dt, r, psi)

    def kuramoto_ata(theta, omega, K, N):
        """
        Right-hand side of the Kuramoto equation using the explicit all-to-all
        pairwise double sum — Eq. (1) of the paper:

            dθ_i/dt = ω_i + (K/N) * Σ_{j=1}^{N} sin(θ_j - θ_i)

        Implementation:
            diff[i,j] = θ_j - θ_i  is formed as the outer difference
                        theta[np.newaxis,:] - theta[:,np.newaxis]
            This is an (N x N) matrix; row i contains (θ_j - θ_i) for all j.
            Taking sin() elementwise and summing over axis=1 (over j) gives
            the coupling term for each oscillator i exactly, with no
            mean-field approximation.

        Complexity: O(N^2) memory and time per step.

        Parameters
        ----------
        theta : (N,) current phases
        omega : (N,) natural frequencies
        K     : float coupling strength
        N     : int  number of oscillators

        Returns
        -------
        dtheta_dt : (N,) time derivatives of phases
        r         : float  order parameter amplitude 
        psi       : float  order parameter phase     
        """
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(np.sin(diff), axis=1)
        dtheta_dt = omega + _K / N * coupling
        r, psi = order_parameter(theta)
        return (dtheta_dt, r, psi)

    def simulate(N, K, omega, theta0, dt, n_steps):
        """
        Euler integration of the Kuramoto model.  # (NxN) matrix: diff[i,j] = theta_j - theta_i
      # sigma_j sin(theta_j - theta_i) for each i
        Parameters
        ----------
        N       : int    number of oscillators
        K       : float  coupling strength
        omega   : (N,)   natural frequencies (fixed, quenched disorder)
        theta0  : (N,)   initial phases
        dt      : float  time step
        n_steps : int    number of integration steps

        Returns
        -------
        theta_hist : (n_steps+1, N)  phase history
        r_hist     : (n_steps+1,)    |order parameter| history
        psi_hist   : (n_steps+1,)    mean phase history
        """
        theta_hist = np.empty((n_steps + 1, N))
        r_hist = np.empty(n_steps + 1)
        psi_hist = np.empty(n_steps + 1)
        theta = theta0.copy()
        theta_hist[0] = theta
        r_hist[0], psi_hist[0] = order_parameter(theta)
        for step in range(1, n_steps + 1):
            drv, r, psi = kuramoto_ata(theta, omega, _K, N)
            theta = theta + dt * drv
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            theta_hist[step] = theta
            r_hist[step] = r
            psi_hist[step] = psi
        return (theta_hist, r_hist, psi_hist)

    def init_circ():
        scat_pts.set_offsets(np.empty((0, 2)))
        r_line.set_data([], [])
        centroid_dot.set_data([], [])
        return (scat_pts, r_line, centroid_dot)
      # Euler step
    def update_circ(frame_idx):  # wrap to (−π,π]
        step = list(frames_circ)[frame_idx]
        theta = th_hist[step]
        r_val = r_hist_d[step]
        psi_v = psi_hist_d[step]
        xs = np.cos(theta)
        ys = np.sin(theta)
        scat_pts.set_offsets(np.column_stack([xs, ys]))
        scat_pts.set_color(osc_colors)
        rx, ry = (r_val * np.cos(psi_v), r_val * np.sin(psi_v))
        arrow_op.set_position((rx, ry))
        arrow_op.xy = (rx, ry)
        arrow_op.xytext = (0.0, 0.0)
        centroid_dot.set_data([rx], [ry])
        t_now = step * DT
        r_line.set_data(time_axis[:step + 1], r_hist_d[:step + 1])
        time_text_c.set_text(f't = {t_now:.1f} s')
        r_text_c.set_text(f'r = {r_val:.3f}')
        return (scat_pts, r_line, centroid_dot)
      # Positions on unit circle
    def init2():
        for ln in lines2.values():
            ln.set_data([], [])
        return list(lines2.values())

    def update2(frame_idx):  # Order parameter arrow
        step = list(frames2)[frame_idx]
        for _K, ln in lines2.items():
            _, r_h, _ = trajs[_K]
            ln.set_data(time_axis[:step + 1], r_h[:step + 1])
        time_txt2.set_text(f't = {step * DT:.1f} s')
        return list(lines2.values())
      # Traces
    def init3():
        th_line.set_data([], [])
        sim_scat.set_offsets(np.empty((0, 2)))
        current_vline.set_xdata([0])
        return (th_line, sim_scat, current_vline)

    def update3(frame_idx):
        k_now = K_SWEEP[frame_idx]
        mask = K_th <= k_now
        th_line.set_data(K_th[mask], r_th[mask])
        sim_scat.set_offsets(np.column_stack([K_SWEEP[:frame_idx + 1], bif_r[:frame_idx + 1]]))
        current_vline.set_xdata([k_now])
        label = 'INCOHERENT' if k_now < K_CRITICAL else 'SYNCHRONISED'
        status_txt.set_text(f'K = {k_now:.2f}   →   {label}')
        return (th_line, sim_scat, current_vline)  # Theory line up to current K  # Simulation dots up to current K

    return (
        init2,
        init3,
        init_circ,
        sample_lorentzian,
        simulate,
        update2,
        update3,
        update_circ,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Critical Coupling $K_c$

    There is a sharp threshold below which the oscillators cannot synchronise as $t \rightarrow \inf$, and above which synchronisation spontaneously emerges. This critical coupling is given by:

    $$K_c = \frac{2}{\pi \, g(0)} = 2\gamma$$

    With $\gamma$ = 0.5, we get $K_c = 1.0$. For $K < K_c$ the incoherent state is stable; for $K > K_c$ a synchronised state bifurcates continuously from zero — a second-order (continuous) phase transition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the Lorentzian distribution, the steady-state order parameter can be solved exactly using contour integration:

    $$r_\infty = \sqrt{1 - \frac{K_c}{K}}, \qquad K > K_c$$

    and $r = 0$ for $K \leq K_c$. As $K \rightarrow K_c$ from above, $r \rightarrow 0$ smoothly. As $K \rightarrow \inf$, $r \rightarrow 1$, though the heavy tails of the Lorentzian mean perfect synchronisation is never quite reached at finite coupling.

    This simulation is performed using the Forward Euler Integration method.
    """)
    return


@app.cell
def _(
    DT,
    GAMMA,
    K_CRITICAL,
    K_SWEEP,
    K_VALUES,
    N,
    N_STEPS,
    RNG_SEED,
    np,
    sample_lorentzian,
    simulate,
):
    # ─────────────────────────────────────────────────────────────
    #  PRE-COMPUTE ALL TRAJECTORIES
    print('Pre-computing trajectories …')
    rng = np.random.default_rng(RNG_SEED)
    omega = sample_lorentzian(N, GAMMA, rng)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    trajs = {}
    for _K in K_VALUES:
    # Initial phases drawn uniformly – fully incoherent start
        trajs[_K] = simulate(N, _K, omega, theta0, DT, N_STEPS)
        print(f'  K = {_K:.2f} --> r_inf ≈ {trajs[_K][1][-200:].mean():.3f}')
    # Dictionary:  K  →  (theta_hist, r_hist, psi_hist)
    print('Pre-computing bifurcation sweep …')
    bif_r = []
    for _K in K_SWEEP:
        _, r_h, _ = simulate(N, _K, omega, theta0, DT, N_STEPS)
        bif_r.append(r_h[int(0.8 * N_STEPS):].mean())
    # Bifurcation sweep: for each K, run until steady state
    bif_r = np.array(bif_r)
    K_th = np.linspace(K_CRITICAL, K_SWEEP[-1], 300)
    r_th = np.sqrt(1.0 - K_CRITICAL / K_th)
    time_axis = np.arange(N_STEPS + 1) * DT
    # Theoretical prediction for Lorentzian: r = sqrt(1 - K_c/K) for K > K_c
    print('All trajectories ready.\n')  # Take mean of last 20 % of simulation as the "steady-state" r
    return K_th, bif_r, r_th, time_axis, trajs


@app.cell
def _(
    Circle,
    DT,
    GAMMA,
    K_CRITICAL,
    Line2D,
    N,
    N_STEPS,
    RNG_SEED,
    T_TOTAL,
    animation,
    file,
    init_circ,
    np,
    os,
    plt,
    sample_lorentzian,
    simulate,
    update_circ,
):
    # ─────────────────────────────────────────────────────────────
    #  ANIMATION 1: Phase oscillators on the unit circle
    #               Shows one coupling
    print('Building Animation 1: oscillators on unit circle …')
    K_demo = 2.0
    rng2 = np.random.default_rng(RNG_SEED)
    omega_d = sample_lorentzian(N, GAMMA, rng2)  # K above K_c to show synchronisation
    theta0_d = rng2.uniform(-np.pi, np.pi, N)
    th_hist, r_hist_d, psi_hist_d = simulate(N, K_demo, omega_d, theta0_d, DT, N_STEPS)
    STRIDE = 4
    frames_circ = range(0, N_STEPS + 1, STRIDE)
    n_frames_c = len(frames_circ)
    # Downsample for smoother animation: every 4 frames
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6), facecolor='#0d0d1a')
    fig1.suptitle(f'Kuramoto Model  —  Phase Oscillators on the Unit Circle\nN = {N},  K = {K_demo:.1f},  K$_c$ = {K_CRITICAL:.2f},  Lorentzian g($\\gamma$),  $\\gamma$ = {GAMMA}', color='white', fontsize=12, y=0.98)
    ax_circ = axes1[0]
    ax_r = axes1[1]
    for ax in axes1:
        ax.set_facecolor('white')
        for _spine in ax.spines.values():
            _spine.set_edgecolor('#333333')
    ax_circ.set_xlim(-1.35, 1.35)
    ax_circ.set_ylim(-1.35, 1.35)
    ax_circ.set_aspect('equal')
    ax_circ.set_title('Oscillator Phases', color='black', fontsize=11)
    ax_circ.tick_params(colors='#666666')
    circle_patch = Circle((0, 0), 1.0, fill=False, edgecolor='#999999', linewidth=1.2, linestyle='--')
    ax_circ.add_patch(circle_patch)
    ax_circ.axhline(0, color='#cccccc', lw=0.6)
    ax_circ.axvline(0, color='#cccccc', lw=0.6)
    ax_circ.text(0.02, 1.25, 'Im', color='#666666', fontsize=9, ha='center')
    ax_circ.text(1.28, 0.02, 'Re', color='#666666', fontsize=9)
    # ---- left panel: unit circle ----
    freq_rank = np.argsort(np.argsort(omega_d))
    osc_colors = plt.cm.plasma(freq_rank / (N - 1))
    scat_pts = ax_circ.scatter([], [], s=50, zorder=5)
    arrow_op = ax_circ.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='#d62728', lw=2.5, mutation_scale=18))
    centroid_dot, = ax_circ.plot([], [], 'o', color='#d62728', markersize=8, zorder=10)
    time_text_c = ax_circ.text(-1.3, -1.28, '', color='black', fontsize=9)
    # Draw the unit circle outline
    r_text_c = ax_circ.text(-1.3, 1.22, '', color='#d62728', fontsize=9, fontweight='bold')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.plasma(0.0), markersize=7, label='Low $\\omega$'), Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.plasma(0.5), markersize=7, label='Medium $\\omega$'), Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.plasma(1.0), markersize=7, label='High $\\omega$'), Line2D([0], [0], color='#d62728', lw=2.5, label='Order param. r·e^{i$\\psi$}')]
    ax_circ.legend(handles=legend_elements, loc='lower right', facecolor='white', edgecolor='#333333', labelcolor='black', fontsize=7.5)
    ax_r.set_xlim(0, T_TOTAL)
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.set_xlabel('Time  t', color='black', fontsize=10)
    ax_r.set_ylabel('Order parameter  r(t)', color='black', fontsize=10)
    ax_r.set_title('Synchronisation Amplitude', color='black', fontsize=11)
    # Colour-map for oscillators: hue = natural frequency rank
    ax_r.tick_params(colors='#666666')  # rank 0 … N-1
    ax_r.axhline(y=np.sqrt(1 - K_CRITICAL / K_demo), color='#2ca02c', linestyle='--', lw=1.2, label=f'Theoretical $r_\\inf$ = {np.sqrt(1 - K_CRITICAL / K_demo):.2f}')
    ax_r.axhline(y=0, color='#cccccc', lw=0.6)
    # Scatter plot of oscillators as dots on the circle
    ax_r.legend(facecolor='white', edgecolor='#333333', labelcolor='black', fontsize=8, loc='upper left')
    r_line, = ax_r.plot([], [], '-', color='#d62728', lw=1.8, label='r(t)')
    # Order parameter arrow (from origin to r·e^{iψ})
    fig1.set_facecolor('white')
    anim1 = animation.FuncAnimation(fig1, update_circ, init_func=init_circ, frames=n_frames_c, interval=40, blit=False)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = str(file) + './outputs/kuramoto/kuramoto_circle.gif'
    _out_dir = os.path.dirname(out1)
    # Centroid dot
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    print('  Saving …')
    anim1.save(out1, writer='pillow', fps=25, dpi=100)
    plt.close(fig1)
    # Legend for colour coding
    # ---- right panel: r(t) trace ----
    print(f'  --> {out1}')
    return (
        arrow_op,
        centroid_dot,
        frames_circ,
        osc_colors,
        psi_hist_d,
        r_hist_d,
        r_line,
        r_text_c,
        scat_pts,
        th_hist,
        time_text_c,
    )


@app.cell
def _(
    GAMMA,
    K_COLORS,
    K_CRITICAL,
    K_VALUES,
    N,
    N_STEPS,
    T_TOTAL,
    animation,
    file,
    init2,
    np,
    os,
    plt,
    update2,
):
    # ─────────────────────────────────────────────────────────────
    #  ANIMATION 2: r(t) for multiple K values simultaneously
    #               Demonstrates the transition through K_c
    print('Building Animation 2: r(t) for multiple K values …')
    STRIDE2 = 4
    frames2 = range(0, N_STEPS + 1, STRIDE2)
    n_frames2 = len(frames2)
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#ffffff')
    ax2.set_facecolor('#ffffff')
    for _spine in ax2.spines.values():
        _spine.set_edgecolor('#333333')
    ax2.set_xlim(0, T_TOTAL)
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel('Time  t', color='black', fontsize=12)
    ax2.set_ylabel('Order parameter  r(t)', color='black', fontsize=12)
    ax2.set_title(f'Synchronisation Amplitude r(t) for Different Coupling Strengths K\nN = {N},  Lorentzian g($\\omega$) with $\\gamma$ = {GAMMA},  K_c = {K_CRITICAL:.2f}', color='black', fontsize=11)
    ax2.tick_params(colors='#666666')
    ax2.axvline(0, color='#d50e0e', lw=0.6)
    ax2.axhline(0, color='#cc0e0e', lw=0.6)
    ax2.axhline(1, color='#b80b0b', lw=0.6)
    lines2 = {}
    for _K, col in zip(K_VALUES, K_COLORS):
        lbl = f'K = {_K:.1f}' + (' (= K_c)' if abs(_K - K_CRITICAL) < 0.05 else '')
        ln, = ax2.plot([], [], '-', color=col, lw=2.0, label=lbl)
        lines2[_K] = ln
    # Shade sub-critical and super-critical regions
    for _K, col in zip(K_VALUES, K_COLORS):
        if _K > K_CRITICAL:
            r_inf = np.sqrt(1.0 - K_CRITICAL / _K)
            ax2.axhline(r_inf, color=col, linestyle=':', lw=1.0, alpha=0.6)
    ax2.axvline(0, color='black', lw=0)
    t_kc = ax2.text(0.5, 0.45, f' K_c = {K_CRITICAL:.2f} separates\n   incoherent (low K) from\n   synchronised (high K)', color='#5353e0', fontsize=8.5, transform=ax2.transAxes, va='center')
    ax2.legend(facecolor='#ffffff', edgecolor='#333333', labelcolor='black', fontsize=9, loc='upper left')
    time_txt2 = ax2.text(0.98, 0.03, '', transform=ax2.transAxes, ha='right', color='black', fontsize=9)
    anim2 = animation.FuncAnimation(fig2, update2, init_func=init2, frames=n_frames2, interval=40, blit=False)
    fig2.tight_layout()
    # Theoretical asymptotic r\infinity markers
    out2 = str(file) + './outputs/kuramoto/kuramoto_multi_K.gif'
    _out_dir = os.path.dirname(out2)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    print('  Saving …')
    # Annotate K_c
    anim2.save(out2, writer='pillow', fps=25, dpi=100)
    plt.close(fig2)
    print(f'  --> {out2}')
    return frames2, lines2, time_txt2


@app.cell
def _(
    GAMMA,
    K_CRITICAL,
    K_SWEEP,
    K_th,
    N,
    animation,
    file,
    init3,
    os,
    plt,
    r_th,
    update3,
):
    # ─────────────────────────────────────────────────────────────
    #  ANIMATION 3: Bifurcation diagram sweep  r_inf  vs  K
    #               Compares simulation to Kuramoto's exact result
    print('Building Animation 3: bifurcation diagram …')
    n_frames3 = len(K_SWEEP)
    fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor='white')
    ax3.set_facecolor('white')
    for _spine in ax3.spines.values():
        _spine.set_edgecolor('#333333')
    ax3.set_xlim(-0.1, K_SWEEP[-1] + 0.1)
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_xlabel('Coupling strength  $K$', color='black', fontsize=12)
    ax3.set_ylabel('Steady-state order parameter $\\langle r \\rangle$', color='black', fontsize=12)
    ax3.set_title('Synchronisation Transition: Bifurcation Diagram' + f'\n$N$ = {N} oscillators,  Lorentzian $g(\\omega)$,  $\\gamma$ = {GAMMA},  $K_c$ = {K_CRITICAL:.2f}', color='black', fontsize=11)
    ax3.tick_params(colors='#666666')
    ax3.axvline(K_CRITICAL, color='#e74c3c', linestyle='--', lw=1.4, label=f'$K_c$ = {K_CRITICAL:.2f}  (= $2\\gamma$)')
    ax3.axhline(0, color='#cccccc', lw=0.6)
    ax3.plot(K_th, r_th, '--', color='#27ae60', lw=1.8, alpha=0.6)
    th_line, = ax3.plot([], [], '-', color='#27ae60', lw=2.2, label='Theory: $r=\\sqrt{1-K_c/K}$')
    sim_scat = ax3.scatter([], [], s=60, color='#3498db', zorder=5, label='Simulation $\\langle r \\rangle$')
    current_vline = ax3.axvline(0, color='#f39c12', lw=1.2, alpha=0.7)
    ax3.fill_betweenx([0, 1.1], 0, K_CRITICAL, color='#ecf0f1', alpha=0.5, label='Incoherent phase')
    ax3.fill_betweenx([0, 1.1], K_CRITICAL, K_SWEEP[-1], color='#d5f4e6', alpha=0.5, label='Synchronised phase')
    ax3.legend(facecolor='white', edgecolor='#333333', labelcolor='black', fontsize=9, loc='upper left')
    status_txt = ax3.text(0.98, 0.05, '', transform=ax3.transAxes, ha='right', color='black', fontsize=10)
    anim3 = animation.FuncAnimation(fig3, update3, init_func=init3, frames=n_frames3, interval=120, blit=False)
    fig3.tight_layout()
    # Plot full theoretical curve in background (faint)
    out3 = str(file) + './outputs/kuramoto/kuramoto_bifurcation.gif'
    # Animated version of the same
    _out_dir = os.path.dirname(out3)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    # Simulation scatter (grown dot by dot)
    print('  Saving …')
    anim3.save(out3, writer='pillow', fps=8, dpi=100)
    plt.close(fig3)
    # Indicator vertical line for current K
    print(f'  --> {out3}')
    print('\nAll animations saved successfully.')
    return current_vline, sim_scat, status_txt, th_line


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Three Animation Visualizations

    This simulation generates three complementary animations saved as GIFs:

    ![Kuramoto Phase Oscillations Animation](public/kuramoto/kuramoto_circle.gif)

    **Animation 1: Phase Oscillators on the Unit Circle**
    - Shows how individual oscillator phases evolve over time on the complex plane
    - Left panel displays oscillators as colored dots on the unit circle, with a red arrow indicating the order parameter $r \cdot e^{i\psi}$
    - Right panel shows the time evolution of the synchronisation amplitude $r(t)$
    - Color coding represents the natural frequency rank (low to high frequencies)

    **Animation 2: Synchronisation Dynamics Across Multiple Coupling Strengths**

    ![Kuramoto Multi-K Animation](public/kuramoto/kuramoto_multi_K.gif)

    This animation demonstrates how the order parameter $r(t)$ evolves differently depending on the coupling strength $K$:
    - **$K = 0.3$ (red, subcritical)**: Remains incoherent with $r \approx 0$ throughout
    - **$K = 0.8$ (orange, near critical)**: Shows weak oscillations around $r < K_c$
    - **$K = 1.0$ (green, at critical point)**: The bifurcation threshold where synchronisation begins to emerge
    - **$K = 1.5$ (blue, supercritical)**: Clear growth toward synchronisation with $r \approx 0.577$
    - **$K = 2.5$ (purple, strongly supercritical)**: Rapid synchronisation reaching $r \approx 0.775$

    The shaded background distinguishes the incoherent phase (left) from the synchronised phase (right) relative to $K_c = 1.0$.

    ![Kuramoto Bifurcation Animation](public/kuramoto/kuramoto_bifurcation.gif)

    **Animation 3: Bifurcation Diagram — Theory vs. Simulation**
    - Sweeps coupling strength from $K = 0$ to $K = 3.0$
    - Green line shows the exact theoretical prediction: $r_\infty = \sqrt{1 - K_c/K}$ for $K \geq K_c$
    - Blue dots accumulate in real time, showing simulation results
    - Vertical line tracks the current coupling strength
    - Demonstrates the sharp second-order phase transition at $K_c$ and excellent agreement between simulation and Kuramoto's exact solution
    """)
    return


if __name__ == "__main__":
    app.run()
