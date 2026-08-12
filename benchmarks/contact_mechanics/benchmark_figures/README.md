# Induced fault slip — benchmark reproduction

Figures in this folder reproduce Novikov, Shokrollahzadeh Behbahani, Voskov,
Hajibeygi & Jansen (2024), *Benchmarking numerical simulation of induced
fault slip with semi-analytical solutions*, Geomech. Geophys. Geo-energ.
Geo-resour. **10**:182, using the **mimetic-AFW-BDM six-field method** of
this library, on the paper's own grids (its Table 3: 2 m cells on the fault,
100 m at the boundary), against the paper's published semi-analytical
dataset (4TU, doi 10.4121/d77f1a2c-29ea-4572-ad72-e33ed8dc8d22).

Regenerate everything with

```
python -m benchmarks.contact_mechanics.reproduce_benchmarks_induced_fault_slip
```

---

## The mimetic-AFW-BDM method with six fields

### Continuum problem

Quasi-static poroelasticity with a frictional fault
$\Gamma_f$. With $\sigma$ the total stress, $p$ the (prescribed) pore
pressure, $\alpha$ the Biot coefficient and $A$ the drained compliance,

$$
A\,\bigl(\sigma + \alpha p\, I\bigr) = \varepsilon(u),\qquad
\nabla\cdot\sigma + f = 0,\qquad
\sigma = \sigma^{T},
$$

and on the fault, writing $t = \sigma n$ for the traction,
$t' = t + \alpha p_f\, n$ for the effective one and
$g = [\![u]\!]$ for the displacement jump in the facet frame
$(n, \tau)$,

$$
g_n \ge 0,\quad t'_n \le 0,\quad g_n\, t'_n = 0,
\qquad
|t'_\tau| \le -\mu\, t'_n,\quad
t'_\tau\,\dot g_\tau = -\mu\, t'_n\, |\dot g_\tau| .
$$

### The building block: the mimetic-BDM stress row

Each row of the stress tensor is discretised by one copy of a
**facet-moment space**: on every facet $e$ of every cell $E$, the degrees of
freedom of a vector field $v$ are the moments of its normal component
against the facet $P_1$ basis $\{\chi_0=1,\ \chi_a = s_a/\sqrt{|e|}\}$,

$$
m_{e,b}(v) \;=\; \int_e (v\cdot n_e)\,\chi_b ,\qquad b = 0,\dots,d-1 ,
$$

i.e. $d$ moments per facet per row, $d^2$ stress DOFs per facet in total.
The local inner product is **consistency-only**: with $N$ the matrix of
those moments applied to a basis of local fields and $G$ their exact Gram
computed against the material tensor,

$$
M_E \;=\; N^{-T} G\, N^{-1},
$$

which enforces the mimetic consistency identities exactly and contains **no
stabilisation term** — there is no free parameter anywhere. Unisolvence of
$N$ on a general polytope is obtained by enriching the $P_1$ modes with
divergence-free **curl-type modes** of increasing potential degree (the
exterior-calculus enrichment); on **simplices no enrichment is needed and
the space and product coincide with $BDM_1$**, degree of freedom for degree
of freedom. The full weakly-symmetric compliance is then $d$ copies of this
row product coupled only through the trace — which is precisely the
structural content of the Arnold–Falk–Winther element: on simplicial meshes
the method **is** AFW, entry by entry; on polytopes it is its natural
mimetic extension.

### The four bulk fields

The compliance splits into deviatoric and volumetric parts,
$A = A_{\mathrm{dev}} + \tfrac{a}{d}\,(\operatorname{tr}\cdot)\, I$, and the
volumetric part is carried by an explicit **solid pressure**
$p_s = \operatorname{tr}_h \sigma / d$ through a $(\Gamma, B)$ pair rather
than being folded into $M$. With the cellwise displacement $u$ and the
cellwise **rotation multiplier** $s$ enforcing symmetry weakly (paired with
the tractions through the facet coordinate expansions $X$), the discrete
saddle problem reads

$$
\begin{pmatrix}
M_{\mathrm{dev}} & \Gamma^{T} & D^{T} & A_s^{T}\\
\Gamma & B & & \\
D & & & \\
A_s & & &
\end{pmatrix}
\begin{pmatrix} \sigma \\ p_s \\ u \\ s \end{pmatrix}
=
\begin{pmatrix} b_\sigma - C^{T} p \\ 0 \\ b_u \\ 0 \end{pmatrix},
$$

where $D$ is the (purely topological) discrete divergence, $A_s$ the
rotation pairing, and $C^{T}p$ the Biot coupling of the prescribed pore
pressure — the only place the flow enters. By exact congruence this
four-field system has the **same solution** as the three-field AFW system;
what changes is the sparsity: the volumetric rank-one updates never fill in
$M$. Traction boundary conditions are *essential* (the traction is a DOF);
displacement conditions are natural.

### The two fracture fields

The fault carries the remaining two fields of the six:

* the **fault traction** $\lambda$ — the fault facets' own stress moments,
  pinned by the contact iteration, so no new unknown type is introduced;
* the **displacement jump** $g$ — read from the residual of the replaced
  constitutive rows of the *unfractured* operator,
  $\,g = -\bigl(M\sigma + D^{T}u + A_s^{T}s - b_f\bigr)_{\Gamma_f}$,
  whose coefficients are exactly the expansion of $[\![u]\!]$ in the facet
  $P_1$ basis — the jump is **piecewise linear per facet**, twice the
  resolution of a collocated scheme on the same mesh.

The contact conditions close the system through the augmented-Lagrangian
fixed point with the Alart–Curnier projection $P$ onto the friction cone,

$$
\lambda \;=\; P\bigl(\lambda + r\,\Delta g\bigr),
$$

driven by a semismooth Newton method on the exactly condensed map
$g(\lambda) = g_0 + \hat G\,\lambda$ (one factorisation of the bulk system;
every contact iteration is then a dense solve of the size of the fault).
The in-situ stress enters as a prestress shift inside $P$, and the Coulomb
threshold acts on the effective traction with the fault pore pressure taken
from the depleted side of the fault.

**The six fields:** $\;(\sigma,\; p_s,\; u,\; s)\;$ in the bulk and
$\;(\lambda,\; g)\;$ on the fault.

---

## Reproduced figures

| figure | content | agreement with the semi-analytical reference |
|---|---|---|
| `benchmark_0.png` | Fig. 4 — unfaulted reservoir, combined stresses on the 70° line | closed forms to plotting accuracy |
| `benchmark_1.png` | Fig. 6 — vertical frictionless displaced fault, Coulomb stress and slip | slip within 0.3 % of the analytic profile |
| `benchmark_2_fig3.png` | Fig. 3 — initial poroelastic state on the 70° line | rms 0.01–0.05 MPa (the linear field is contained in the stress space) |
| `benchmark_2_fig8.png` | Fig. 8 — pre-slip stresses at $p=-25$ MPa | rms 0.12–0.34 MPa on 9–31 MPa scales |
| `benchmark_2_fig9.png` | Fig. 9 — post-slip state, $W = 4500$ m | $\Sigma_C$ rms 0.10 MPa; slip rms 0.36 mm |
| `benchmark_2_fig10.png` | Fig. 10 — post-slip state, $W = 18{,}000$ m | $\Sigma_C$ rms 0.10 MPa; slip rms 0.39–0.57 mm |
| `benchmark_2_fig12.png` | Fig. 12 — slip-patch boundaries and merging | merging at $-26.90$ vs $-26.87$ MPa; boundaries on the surviving reference rows (upper rows are lost in the published dataset and shown mirrored) |
| `benchmark_3_fig14.png` | Fig. 14 — slip-weakening pre-nucleation state | slip rms 0.20 mm; $p^{*} \in (-17.13, -17.11)$ MPa vs $-17.27$ (DARTS) / $-17.41$ (semi-analytical) |

Known defects of the published 4TU dataset (verified element-wise) are
documented in `../reference_data.py`; comparisons above use only genuine
rows, plus the reconstructed exact slip of Figs. 9–10 obtained by solving
the governing Cauchy singular integral equation with the genuine stress
rows as data.
