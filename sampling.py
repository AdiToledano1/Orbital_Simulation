"""
Sampling: draw points distributed like |psi|^2 (replaces baked PNGs/grids)
============================================================================

Three samplers, all reusing the same exact single-atom sampler as their
foundation:

  * sample_orbital               -- one hydrogen-like (n,l,m) orbital, exactly.
  * sample_lcao                  -- two-center LCAO (see its section below).
  * sample_at_tau (+ reservoir)  -- an arbitrary superposition of same-nucleus
                                     (n,l,m) eigenstates, time-evolved via
                                     exp(-i E_n tau) (see its section below).

The first two are the "expensive" step in this project, and every tab only
ever calls its resampler when the physical state actually changed (n, l, m,
Z, distance, bonding sign, or which terms/coefficients are chosen) -- never
on camera moves or display-only controls (samples-shown, quantity, colormap,
target/delta, cut-away). The superposition sampler splits into an expensive
tau-independent half (build once) and a cheap tau-dependent half (rerun on
every tau slider move) -- see its section below for why that split is exact,
not an approximation.
"""

from functools import lru_cache

import numpy as np
from scipy import special

from physics import A0, radial_R, psi_nlm

# ============================================================================
# Single-atom sampler
# ============================================================================
#
# |psi_nlm|^2 dV separates as [ r^2 R_nl(r)^2 dr ] * [ |Y_l^m(theta)|^2 sin(theta) dtheta ] * [ dphi / 2pi ]
# because |Y_l^m(theta,phi)|^2 does not depend on phi. That means:
#   - phi is sampled exactly and uniformly on [0, 2pi)
#   - r is sampled exactly via inverse-CDF lookup on a precomputed radial grid
#   - theta is sampled via 1D rejection sampling (fast and robust for every l, m)
# This is both exact (no discretization error beyond grid resolution) and
# fast even for the most sharply peaked orbitals (e.g. n=7, l=6, m=6 draws
# 20,000 points in ~25ms).

@lru_cache(maxsize=64)
def _radial_cdf(n, l, a0=A0, n_grid=20000):
    # The margin has to scale with a0 too (not just the n^2 term), or a
    # small a0_eff (high nuclear charge Z -- see a0_for_z) leaves almost
    # all of this grid's resolution wasted on empty tail far past where
    # the orbital actually has any density, and the inverse-CDF sampling
    # below gets numerically coarse right where it matters most.
    r_max = a0 * (6.0 * n ** 2 + 20.0)
    r = np.linspace(1e-8, r_max, n_grid)
    dens = r ** 2 * radial_R(r, n, l, a0) ** 2
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    return r, cdf


@lru_cache(maxsize=64)
def _angular_theta_grid(l, m, n_grid=4000):
    theta_grid = np.linspace(0, np.pi, n_grid)
    Y = special.sph_harm_y(l, m, theta_grid, 0.0)
    dens = np.abs(Y) ** 2 * np.sin(theta_grid)
    return theta_grid, dens


@lru_cache(maxsize=64)
def sample_orbital_cached(n, l, m, n_points, a0=A0, seed=0, max_guard=500):
    """Cached wrapper around sample_orbital: revisiting an orbital (e.g. via
    the arrow keys) is then free -- no resampling, no waiting."""
    rng = np.random.default_rng(seed)
    return sample_orbital(n, l, m, n_points, a0=a0, rng=rng, max_guard=max_guard)


def sample_orbital(n, l, m, n_points, a0=A0, rng=None, max_guard=500):
    """Return (points[N,3], psi_values[N]) sampled ~ |psi_nlm|^2."""
    rng = rng if rng is not None else np.random.default_rng()

    r_grid, cdf_r = _radial_cdf(n, l, a0)
    u = rng.random(n_points)
    r = np.interp(u, cdf_r, r_grid)

    theta_grid, ang_dens_grid = _angular_theta_grid(l, m)
    ang_max = ang_dens_grid.max() * 1.2

    theta = np.empty(n_points)
    filled = 0
    guard = 0
    while filled < n_points and guard < max_guard:
        need = n_points - filled
        cand = rng.uniform(0, np.pi, size=need * 3)
        Yc = special.sph_harm_y(l, m, cand, 0.0)
        dens = np.abs(Yc) ** 2 * np.sin(cand)
        u2 = rng.uniform(0, ang_max, size=len(cand))
        acc = cand[u2 < dens]
        take = min(len(acc), need)
        theta[filled:filled + take] = acc[:take]
        filled += take
        guard += 1
    if filled < n_points:  # extremely unlikely fallback
        p = ang_dens_grid / ang_dens_grid.sum()
        theta[filled:] = rng.choice(theta_grid, size=n_points - filled, p=p)

    phi = rng.uniform(0, 2 * np.pi, size=n_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    psi_vals = psi_nlm(x, y, z, n, l, m, a0)
    return np.column_stack([x, y, z]), psi_vals


# ============================================================================
# LCAO - two-center linear combination of atomic orbitals
# ============================================================================
#
#   Psi(r) = psi_nlm(r - R_A) +/- psi_nlm(r - R_B)
#
# with the same (n, l, m) orbital on each nucleus, nucleus A at z=-d/2 and
# nucleus B at z=+d/2 ("+" = bonding, "-" = antibonding). This is the
# standard textbook homonuclear-diatomic LCAO picture (sigma/pi bonding and
# antibonding combinations of s, p, d, ... orbitals).
#
# To sample points ~ |Psi|^2 without a slow generic 3D grid-rejection
# scheme, this reuses the exact single-atom sampler above as an envelope:
# propose each point from a 50/50 mixture of "centered on A" and "centered
# on B" (both exactly sampled via sample_orbital, just translated), then
# accept with probability |Psi|^2 / (4 * envelope), where
# envelope = 0.5*(|psi_A|^2 + |psi_B|^2). Because
# |Psi|^2 = |psi_A|^2 + |psi_B|^2 +/- 2 Re(psi_A* psi_B)
# and |Re(psi_A* psi_B)| <= |psi_A||psi_B| <= 0.5(|psi_A|^2+|psi_B|^2)
# (AM-GM), that acceptance ratio never exceeds 1, so this is an exact,
# unbiased rejection sampler that is always fast (empirically one pass is
# enough -- see sample_lcao's guard loop) and needs no new physics beyond
# what sample_orbital / psi_nlm already compute.

def sample_lcao(n, l, m, distance, sign, n_points, a0=A0, rng=None, max_guard=60, batch_factor=3.0):
    """Return (points[N,3], psi_values[N]) sampled ~ |psi_A +/- psi_B|^2,
    two same-(n,l,m) hydrogen orbitals centered on the z-axis, `distance`
    apart. sign=+1 -> bonding, sign=-1 -> antibonding."""
    rng = rng if rng is not None else np.random.default_rng()
    half = distance / 2.0

    pts_chunks, psi_chunks = [], []
    have = 0
    guard = 0
    while have < n_points and guard < max_guard:
        need = n_points - have
        batch = max(int(need * batch_factor), 256)
        n_a = int(rng.binomial(batch, 0.5))
        n_b = batch - n_a

        pa, _ = sample_orbital(n, l, m, n_a, a0, rng) if n_a else (np.empty((0, 3)), None)
        pb, _ = sample_orbital(n, l, m, n_b, a0, rng) if n_b else (np.empty((0, 3)), None)
        pa = pa.copy()
        pb = pb.copy()
        if pa.shape[0]:
            pa[:, 2] -= half  # proposals drawn "centered on A" -> shift to A's actual position
        if pb.shape[0]:
            pb[:, 2] += half  # proposals drawn "centered on B" -> shift to B's actual position
        pts = np.vstack([pa, pb])

        psi_a = psi_nlm(pts[:, 0], pts[:, 1], pts[:, 2] + half, n, l, m, a0)  # relative to A at -half
        psi_b = psi_nlm(pts[:, 0], pts[:, 1], pts[:, 2] - half, n, l, m, a0)  # relative to B at +half
        psi_total = psi_a + sign * psi_b

        dens_total = np.abs(psi_total) ** 2
        envelope = 0.5 * (np.abs(psi_a) ** 2 + np.abs(psi_b) ** 2)
        envelope = np.where(envelope < 1e-300, 1e-300, envelope)
        accept_p = np.clip(dens_total / (4.0 * envelope), 0.0, 1.0)
        keep = rng.random(batch) < accept_p

        pts_chunks.append(pts[keep])
        psi_chunks.append(psi_total[keep])
        have += int(keep.sum())
        guard += 1

    points = np.vstack(pts_chunks) if pts_chunks else np.empty((0, 3))
    psi_vals = np.concatenate(psi_chunks) if psi_chunks else np.empty((0,), dtype=complex)
    if points.shape[0] > n_points:
        # IMPORTANT: pick a random subset, not the first n_points. The
        # accepted points are ordered A-batch-first-then-B-batch (see
        # `pts = np.vstack([pa, pb])` above), so a plain prefix slice would
        # systematically keep more of nucleus A's points than nucleus B's
        # whenever more points were accepted than needed (the usual case)
        # -- silently biasing the cloud towards one atom, worse the more
        # lopsided the accept-rate bookkeeping got. The *set* of accepted
        # points is an unbiased sample of |Psi|^2; a uniform random subset
        # of it is too, so shuffle before truncating.
        idx = rng.permutation(points.shape[0])[:n_points]
        points = points[idx]
        psi_vals = psi_vals[idx]
    return points, psi_vals


@lru_cache(maxsize=64)
def sample_lcao_cached(n, l, m, distance, sign, n_points, a0=A0, seed=0):
    """Cached wrapper (same rationale as sample_orbital_cached)."""
    rng = np.random.default_rng(seed)
    return sample_lcao(n, l, m, distance, sign, n_points, a0=a0, rng=rng)


# ============================================================================
# SUPERPOSITION / TIME EVOLUTION - arbitrary (n,l,m) terms, one nucleus
# ============================================================================
#
#   Psi(r, tau) = sum_i c_i * psi_{n_i,l_i,m_i}(r) * exp(-i E_{n_i} tau)
#
# c_i are real (their sign sets the tau=0 relative phase); tau is time in
# atomic units (hbar=1, matching energy_level() in physics.py). This
# generalizes the old same-n s/px/py/pz hybridization feature (still
# reachable as a special case: pick terms that all share one n) and adds
# genuine time dynamics whenever two terms have different n and therefore
# different energy -- see the chat for the full derivation of why that
# combination beats and a same-n one doesn't.
#
# Keeping a manual tau slider responsive hinges on one fact:
# |c_i * exp(-i E_i tau)| = |c_i|, independent of tau. That means both the
# proposal mixture weights AND the rejection-sampling envelope below depend
# only on the c_i^2, never on tau -- so the *expensive* part (drawing
# proposal points and evaluating psi_nlm at each of them) can be done once,
# into a reusable "reservoir", and reused unchanged for every tau. Moving
# the tau slider only re-runs the cheap accept/reject arithmetic over that
# already-computed reservoir -- no psi_nlm evaluation happens on a tau
# change, matching this project's rule that only genuinely structural
# changes (which (n,l,m,c) terms are chosen, or Z) trigger a real resample.
#
# The bound behind the rejection sampler is the complex version of the same
# Cauchy-Schwarz argument used for LCAO/hybrid:
#   |sum_i z_i|^2 <= K * sum_i |z_i|^2   for any K complex numbers z_i
# (apply ordinary Cauchy-Schwarz to the vectors (1,...,1) and (z_1,...,z_K)).
# With z_i = c_i * exp(-i E_i tau) * psi_i(r), |z_i| = |c_i| * |psi_i(r)| is
# tau-independent, so this bound -- and the envelope built from it -- holds
# identically at every tau, which is exactly why one reservoir can serve
# every tau. (Verified numerically before wiring this into the UI: the
# resulting accept probability never exceeds 1 across a wide sweep of tau,
# terms of different n reproduce a z>0/z<0 density oscillation at exactly
# the predicted beat period 2*pi/|E_i-E_j|, and terms that all share one n
# give bit-for-bit identical output at every tau -- degenerate states truly
# don't beat.)
#
# One genuinely interesting (not a bug!) consequence of using the *exact*
# hydrogenic radial functions rather than idealized textbook lobes: for
# n=2, R_20 (the "2s" radial part) has a radial node at r=2*a0 and is
# NEGATIVE beyond it, while R_21 ("2p") has no radial node and stays one
# sign throughout. This still shows up in same-n superpositions (e.g. a
# 2s+2p_z mix) exactly as it did in the old hybridization feature.

def _build_superposition_reservoir(terms, a0=A0, reservoir_size=400_000, rng=None):
    """terms: tuple of (n, l, m, c) with c real, c != 0, at least one term.
    Returns a dict holding everything sample_at_tau() needs to answer any
    tau cheaply: the proposal points, each term's raw (un-coefficiented)
    psi_i value at every point, the (tau-independent) envelope, a fixed
    permutation (so that when more points pass than are needed, *which*
    subset gets shown is stable rather than re-randomized every tau), and a
    fixed uniform threshold per point (so the accept/reject decision itself
    is stable -- a point flips in/out smoothly as tau sweeps continuously,
    instead of the whole cloud "sparkling" on every slider tick)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    K = len(terms)
    c = np.array([t[3] for t in terms], dtype=float)
    c2 = c * c
    total_c2 = c2.sum()
    probs = c2 / total_c2

    counts = rng.multinomial(reservoir_size, probs)
    parts = []
    for (n, l, m, _c), cnt in zip(terms, counts):
        if cnt:
            pts_i, _ = sample_orbital(n, l, m, int(cnt), a0, rng)
            parts.append(pts_i)
    points = np.vstack(parts) if parts else np.empty((0, 3))
    M = points.shape[0]

    raw = np.empty((M, K), dtype=complex)
    for i, (n, l, m, _c) in enumerate(terms):
        raw[:, i] = psi_nlm(points[:, 0], points[:, 1], points[:, 2], n, l, m, a0)

    envelope = (np.abs(raw) ** 2 * c2[None, :]).sum(axis=1) / total_c2
    envelope = np.where(envelope < 1e-300, 1e-300, envelope)

    return {
        "points": points,
        "raw": raw,                    # (M, K) complex, psi_i(r) WITHOUT c_i
        "c": c,                        # (K,) real coefficients
        "K": K,
        "total_c2": total_c2,
        "envelope": envelope,          # (M,) real, tau-independent
        "order": rng.permutation(M),   # fixed subset-selection order
        "u": rng.random(M),            # fixed accept/reject thresholds
    }


def sample_at_tau(reservoir, energies, tau, n_points):
    """The cheap, tau-dependent half: given a reservoir built once by
    _build_superposition_reservoir() and each term's energy E_i, return
    (points[N,3], psi_values[N]) sampled ~ |Psi(r,tau)|^2. No psi_nlm
    evaluation happens here -- just complex arithmetic over already-
    computed values, which is why dragging the tau slider stays fast."""
    raw = reservoir["raw"]
    c = reservoir["c"]
    K = reservoir["K"]
    total_c2 = reservoir["total_c2"]
    envelope = reservoir["envelope"]
    points = reservoir["points"]

    phase = np.exp(-1j * np.asarray(energies, dtype=float) * tau)  # (K,)
    weighted = raw * (c * phase)[None, :]                            # (M, K)
    psi_total = weighted.sum(axis=1)                                  # Psi(r, tau)

    accept_p = np.clip(np.abs(psi_total) ** 2 / (K * total_c2 * envelope), 0.0, 1.0)
    accept_mask = accept_p > reservoir["u"]

    order = reservoir["order"]
    ordered_accepted = order[accept_mask[order]]
    selected = ordered_accepted[:n_points]

    return points[selected], psi_total[selected]


@lru_cache(maxsize=32)
def superposition_reservoir_cached(terms, a0=A0, reservoir_size=400_000, seed=0):
    """Cached wrapper (same rationale as sample_orbital_cached). `terms`
    must be a hashable tuple of (n, l, m, c) tuples, sorted, with c already
    rounded to a fixed precision by the caller (see HybridTab._resample)
    so that visually-identical slider positions reuse the same cache
    entry instead of missing on float noise."""
    rng = np.random.default_rng(seed)
    return _build_superposition_reservoir(terms, a0=a0, reservoir_size=reservoir_size, rng=rng)


# A handful of illustrative starting points: (display name, [(n,l,m,c), ...]).
# Each list has at most 4 entries (the UI offers 4 term slots).
SUPERPOSITION_PRESETS = [
    ("1s + 2p_z  (different n -- beats)", [(1, 0, 0, 1 / np.sqrt(2)), (2, 1, 0, 1 / np.sqrt(2))]),
    ("1s + 2s  (different n -- beats)", [(1, 0, 0, 1 / np.sqrt(2)), (2, 0, 0, 1 / np.sqrt(2))]),
    ("2s + 2p_z  (same n -- static)", [(2, 0, 0, 1 / np.sqrt(2)), (2, 1, 0, 1 / np.sqrt(2))]),
    ("1s + 2p_z + 3d_z2  (3-way beat)", [(1, 0, 0, 0.6), (2, 1, 0, 0.6), (3, 2, 0, 0.6)]),
    ("Pure 1s", [(1, 0, 0, 1.0)]),
    ("Pure 2p_z", [(2, 1, 0, 1.0)]),
]
SUPERPOSITION_PRESET_NAMES = [name for name, _ in SUPERPOSITION_PRESETS]
