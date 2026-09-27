"""Mixed-dimensional linear contact through the Python interface.

The Python-side mirror of tests/model/test_contact_patch.cpp: confined uniaxial
compression with a fracture across the load, on both symmetry families.

A FRACTURE IS A NODE SET, and that is the whole of how one is identified. The
mesh is cut along it when the model is built, so the two sides carry separate
tractions and the stress may jump; a facet index taken before the cut still
means that facet after it, because the cut appends its copies rather than
renumbering.

WHY THE BONDED CASE IS THE TEST. With the compliance zero the interface
transmits everything and opens by nothing, so the cut problem IS the uncut
problem written in more unknowns, and the same closed form must come back to
round-off. It is also the only configuration that pins the SIGNS of the
coupling: no assertion about where matrix entries land can tell T from -T.

Give a stiffness and the field gains a jump of sigma_nn / c_n across the
fracture, which is checked too -- and being statically determinate, the stress
does not move at all.
"""

import numpy as np
import pathlib

import pytest

import mimetika_cxx as mk

MU, LAM = 1.0, 1.0
LOAD = 1.0
AXIS = 2  # confined compression along z, the fracture across it

WEAK = (mk.StressRealization.derham_bdm, mk.StressFormulation.weak_symmetry)
STRONG = (mk.StressRealization.stabilized_vem, mk.StressFormulation.strong_symmetry)
FAMILIES = {"weak": WEAK, "strong": STRONG}


def _cut_box(n, family):
    """The unit cube, n cells a side, and the node set of the plane z = 1/2."""
    mesh = mk.box([n, n, n], 3, family)
    points = np.array([mesh.point(v) for v in range(mesh.count(0))])
    nodes = [int(v) for v in np.flatnonzero(np.abs(points[:, AXIS] - 0.5) < 1e-12)]
    assert nodes, "no node plane at z = 1/2; n must be even"
    return mesh, nodes


def _solve(n, family, which, c_n=0.0, fracture=True):
    how, form = FAMILIES[which]
    mesh, nodes = _cut_box(n, family)
    model = mk.CauchyMechanicsModel(mesh, 3, mk.ElasticMaterial(MU, LAM), how, form)
    if fracture:
        model.fracture([nodes])
        if c_n > 0.0:
            model.linear_contact(c_n, 7.0)
        else:
            model.bond()

    # the boundary of the DOMAIN: the model drops the fracture's own copies,
    # which have one cofacet each and would otherwise pass for boundary facets
    loaded, confined = [], []
    for f in model.outer_boundary_facets():
        x = mk.centroid(mesh, 2, f)
        (loaded if abs(x[AXIS] - 1.0) < 1e-9 else confined).append(f)
    applied = [0.0] * 9
    applied[AXIS * 3 + AXIS] = -LOAD
    model.add_traction(loaded, applied)
    model.add_free_slip(confined)
    model.solve()
    return model, mesh


def _closed_form(model, mesh, c_n):
    """Worst error in u against eps_axial z, plus the jump every cell above
    the fracture has also taken."""
    k_oed = LAM + 2.0 * MU
    e_exact = -LOAD / k_oed
    jump = -LOAD / c_n if c_n > 0.0 else 0.0
    worst = 0.0
    for e in range(model.n_cells):
        z = mk.centroid(mesh, 3, e)[AXIS]
        want = e_exact * z + (jump if z > 0.5 else 0.0)
        worst = max(worst, abs(model.displacement(e, AXIS) - want))
    return worst


@pytest.mark.parametrize("which", sorted(FAMILIES))
@pytest.mark.parametrize("family", [mk.Family.cartesian, mk.Family.simplex])
def test_a_bonded_fracture_is_invisible(which, family):
    """The cut answer is the uncut answer, to round-off, not merely near it."""
    cut, mesh = _solve(2, family, which)
    whole, _ = _solve(2, family, which, fracture=False)
    assert cut.fractured and not whole.fractured
    assert cut.n_fracture_faces > 0
    worst = max(
        abs(cut.displacement(e, k) - whole.displacement(e, k))
        for e in range(cut.n_cells)
        for k in range(3)
    )
    assert worst < 1e-10
    assert _closed_form(cut, mesh, 0.0) < 1e-10


@pytest.mark.parametrize("which", sorted(FAMILIES))
@pytest.mark.parametrize("c_n", [2.0, 20.0])
def test_a_compliant_fracture_closes_by_the_compliance(which, c_n):
    """[[u_n]] = sigma_nn / c_n, and the stress does not move: uniaxial
    compression is statically determinate, so it crosses any closed interface
    whatever its stiffness."""
    model, mesh = _solve(2, mk.Family.cartesian, which, c_n=c_n)
    assert _closed_form(model, mesh, c_n) < 1e-10
    worst_stress = max(
        abs(model.cell_stress(e)[AXIS * 3 + AXIS] + LOAD) for e in range(model.n_cells)
    )
    assert worst_stress < 1e-9


@pytest.mark.parametrize("which", sorted(FAMILIES))
def test_the_fracture_reads_back_in_its_own_frame(which):
    """Per face, in {n, t1, t2}: t_c . n = sigma_nn = -LOAD, the opening is
    negative because compression closes, nothing shears, and the two numbers
    obey the law they came from -- one sign convention, not two."""
    c_n = 4.0
    model, _ = _solve(2, mk.Family.cartesian, which, c_n=c_n)
    fr = model.fracture_faces()
    assert set(fr) == {
        "id", "area", "normal", "tangent1", "tangent2",
        "traction", "u_plus", "u_minus", "jump",
    }
    n = model.n_fracture_faces
    assert all(np.asarray(fr[k]).shape[0] == n for k in fr)
    assert (fr["id"] == 0).all()  # one fracture, and it is labelled as such
    assert abs(fr["area"].sum() - 1.0) < 1e-12  # the unit square at z = 1/2
    assert np.allclose(np.abs(fr["normal"][:, AXIS]), 1.0, atol=1e-12)

    assert np.allclose(fr["traction"][:, 0], -LOAD, atol=1e-10)
    assert np.allclose(fr["jump"][:, 0], -LOAD / c_n, atol=1e-10)
    assert np.allclose(fr["traction"][:, 1:], 0.0, atol=1e-10)
    assert np.allclose(fr["jump"][:, 1:], 0.0, atol=1e-10)
    # the law, in the numbers as reported
    assert np.allclose(fr["traction"][:, 0], c_n * fr["jump"][:, 0], atol=1e-10)
    # and the jump really is the difference of the halves
    assert np.allclose(fr["jump"], fr["u_plus"] - fr["u_minus"], atol=1e-14)


def test_the_frame_is_orthonormal():
    """It is the frame the compliance was built in, so a traction resolved with
    it and a stiffness applied there cannot disagree about which way is normal."""
    model, _ = _solve(2, mk.Family.simplex, "weak", c_n=4.0)
    fr = model.fracture_faces()
    n, t1, t2 = fr["normal"], fr["tangent1"], fr["tangent2"]
    for a in (n, t1, t2):
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)
    for a, b in ((n, t1), (n, t2), (t1, t2)):
        assert np.allclose((a * b).sum(axis=1), 0.0, atol=1e-12)
    # right-handed, so {n, t1, t2} orients the plane the way a reader expects
    assert np.allclose(np.cross(t1, t2), n, atol=1e-12)


# -- several fractures at once -------------------------------------------------
#
# ONE CUT CARRYING ALL OF THEM, and it has to agree face for face with cutting
# each on its own. Not by merging the node sets: a facet belongs to a fracture
# when all of its vertices do, and that test applied to a UNION of node sets
# admits facets belonging to no fracture at all -- on a tetrahedron with one
# vertex on one plane and two on another, the union holds all three. Each
# fracture induces its own facets and the union of THOSE is the stratum.
#
# The planes here cross, which is the case that shows the difference.


def _two_planes(n=2):
    mesh = mk.box([n, n, n], 3, mk.Family.simplex)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    return mesh, [
        [int(v) for v in np.flatnonzero(np.abs(pts[:, axis] - 0.5) < 1e-12)]
        for axis in (0, AXIS)
    ]


# A SHEARED datum, not confined compression: an axis-aligned fracture under a
# purely axial load carries no shear at all, so c_T would go untested and the
# comparison would pass however it was wired. This gradient shears every plane.
SHEAR = [0.30, 0.22, -0.15,
         0.18, -0.25, 0.12,
         -0.10, 0.14, 0.40]


def _solve_sets(mesh, node_sets, c_n, c_t, gradient=None, labels=None):
    model = mk.CauchyMechanicsModel(
        mesh, 3, mk.ElasticMaterial(MU, LAM), *FAMILIES["weak"]
    )
    model.fracture(node_sets, labels or [])
    model.linear_contact(c_n, c_t)
    if gradient is None:
        loaded, confined = [], []
        for f in model.outer_boundary_facets():
            x = mk.centroid(mesh, 2, f)
            (loaded if abs(x[AXIS] - 1.0) < 1e-9 else confined).append(f)
        applied = [0.0] * 9
        applied[AXIS * 3 + AXIS] = -LOAD
        model.add_traction(loaded, applied)
        model.add_free_slip(confined)
    else:
        # u = G (x - x_E) + G x_E on every boundary facet, stated about the
        # cofacet centroid, which is how every other driver states it
        for f in model.outer_boundary_facets():
            xe = mk.centroid(mesh, 3, int(mk.cofacet_of(mesh, 3, f)))
            constant = [sum(gradient[i * 3 + j] * xe[j] for j in range(3)) for i in range(3)]
            model.prescribe_displacement([f], constant, gradient)
    model.solve()
    return model


def test_all_fractures_at_once_matches_one_at_a_time():
    """Face for face, per fracture -- so the union admitted nothing extra."""
    mesh, sets = _two_planes()
    each = [_solve_sets(mesh, [ns], 4.0, 7.0).n_fracture_faces for ns in sets]
    both = _solve_sets(mesh, sets, 4.0, 7.0)
    ids = both.fracture_faces()["id"]
    assert both.n_fracture_faces == sum(each)
    assert [int((ids == i).sum()) for i in range(len(sets))] == each
    # every face is claimed by exactly one fracture, none left over
    assert set(np.unique(ids)) == set(range(len(sets)))


def test_the_two_compliances_act_independently():
    """c_n governs the opening and c_T the slip, and neither leaks into the
    other: the law holds separately in each part."""
    c_n, c_t = 50.0, 5.0
    mesh, sets = _two_planes()
    fr = _solve_sets(mesh, sets, c_n, c_t, SHEAR).fracture_faces()
    t_n, j_n = fr["traction"][:, 0], fr["jump"][:, 0]
    t_t, j_t = fr["traction"][:, 1:], fr["jump"][:, 1:]
    assert np.allclose(t_n, c_n * j_n, atol=1e-10)
    assert np.allclose(t_t, c_t * j_t, atol=1e-10)
    # and both parts are genuinely nonzero, or the checks above are vacuous --
    # which is exactly what an axial load on an axis-aligned plane would make them
    assert np.abs(j_t).max() > 1e-3
    assert np.abs(j_n).max() > 1e-3


def test_the_fracture_vtu_splits_normal_from_tangential(tmp_path):
    r"""The orthogonal split, and nothing else.

    On a face with frame {n, t1, t2} every vector quantity decomposes as
    v = (v.n) n + P_T v with P_T = I - n (x) n, and the two parts are objects of
    different kinds: a SCALAR v.n, and the tangential part as an element of the
    face's 2-dimensional tangent space, components (v.t1, v.t2).

    That is the basis D_c is DIAGONAL in -- 1/c_n on the normal part, 1/c_T
    times the identity on the tangential one -- so the law must come back
    COMPONENT BY COMPONENT, which is what is checked: a magnitude would pass
    even if t1 and t2 had been swapped or mixed.
    """
    meshio = pytest.importorskip("meshio")
    c_n, c_t = 50.0, 5.0
    mesh, sets = _two_planes()
    model = _solve_sets(mesh, sets, c_n, c_t, SHEAR)
    path = tmp_path / "fractures.vtu"
    model.write_fracture_vtu(str(path))

    got = meshio.read(str(path))
    assert sum(len(c.data) for c in got.cells) == model.n_fracture_faces
    flat = {
        k: np.concatenate([np.asarray(b).reshape(len(b), -1) for b in v])
        for k, v in got.cell_data.items()
    }
    # one scalar and d - 1 tangential components per quantity, nothing padded
    # and nothing re-embedded into ambient axes
    for q in ("traction", "jump", "u_c_plus", "u_c_minus"):
        for part in ("_n", "_t1", "_t2"):
            assert flat[q + part].shape[1] == 1, f"{q}{part} is not a scalar component"
        assert f"{q}_t" not in flat  # not a magnitude: that discards the direction
        assert f"{q}_t_vec" not in flat  # nor three numbers for a 2D object
    # the frame, which is what carries a tangential pair back into space
    for k in ("normal", "tangent1", "tangent2"):
        assert flat[k].shape[1] == 3

    # D_c IS DIAGONAL IN THIS BASIS, one entry at a time
    assert np.allclose(flat["traction_n"], c_n * flat["jump_n"], atol=1e-10)
    assert np.allclose(flat["traction_t1"], c_t * flat["jump_t1"], atol=1e-10)
    assert np.allclose(flat["traction_t2"], c_t * flat["jump_t2"], atol=1e-10)
    # and both tangential components carry something, or the last two are vacuous
    assert np.abs(flat["jump_t1"]).max() > 1e-3
    assert np.abs(flat["jump_t2"]).max() > 1e-3

    # the components are the frame's: n . (t1, t2) = 0 and each is a unit vector
    assert np.allclose((flat["normal"] * flat["tangent1"]).sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.cross(flat["tangent1"], flat["tangent2"]), flat["normal"], atol=1e-12)

    # the jump is the difference of the traces, in every component
    for part in ("_n", "_t1", "_t2"):
        assert np.allclose(flat["jump" + part],
                           flat["u_c_plus" + part] - flat["u_c_minus" + part], atol=1e-14)
    assert set(np.unique(flat["fracture_id"])) == set(range(len(sets)))

    # NO ARRAY WRITTEN TWICE. Two DataArrays of one name in a CellData section
    # is not a valid .vtu, and ParaView answers by showing an empty mesh rather
    # than by complaining -- so it is read from the file itself, which meshio
    # would quietly deduplicate.
    import re

    xml = pathlib.Path(path).read_bytes().decode("utf-8", "ignore")
    block = re.search(r"<CellData(.*?)</CellData>", xml, re.S).group(1)
    names = re.findall(r'Name="([^"]+)"', block)
    assert len(names) == len(set(names)), f"duplicated: {[n for n in names if names.count(n) > 1]}"


def test_a_face_reports_the_fracture_it_belongs_to_and_not_its_position():
    """A subset selection must not renumber. Asked for fractures 2 and 7, the
    faces come back as 2 and 7 -- positional ids would call them 0 and 1, and a
    plot would name the wrong plane while every number on it stayed right."""
    mesh, sets = _two_planes()
    model = _solve_sets(mesh, sets, 50.0, 5.0, SHEAR, labels=[2, 7])
    ids = model.fracture_faces()["id"]
    assert set(np.unique(ids)) == {2, 7}
    # and they sit on the faces the unlabelled run put them on
    plain = _solve_sets(mesh, sets, 50.0, 5.0, SHEAR).fracture_faces()["id"]
    assert np.array_equal(np.where(ids == 2, 0, 1), plain)


def test_labels_must_match_the_node_sets():
    mesh, sets = _two_planes()
    model = mk.CauchyMechanicsModel(
        mesh, 3, mk.ElasticMaterial(MU, LAM), *FAMILIES["weak"]
    )
    with pytest.raises(Exception):
        model.fracture(sets, [1])


def test_the_paraview_reconstruction_gives_the_ambient_vector(tmp_path):
    r"""v_n * n + v_t1 * t1 + v_t2 * t2 is the vector back in ambient axes.

    Checked against an answer that does not come from the frame at all. Under
    confined compression across the plane z = 1/2 the traction on the fracture
    is sigma . n = (0, 0, -LOAD) and the opening is (0, 0, -LOAD/c_n), both in
    AMBIENT coordinates and both known before anything is discretized. So if the
    frame written to the file were not the frame the components were resolved in
    -- a swapped tangent, a flipped normal, a tangent2 from a different rule --
    the reconstruction would not land on them.

    This is the expression to type into a ParaView Calculator.
    """
    meshio = pytest.importorskip("meshio")
    c_n = 4.0
    mesh = mk.box([2, 2, 2], 3, mk.Family.simplex)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    nodes = [int(v) for v in np.flatnonzero(np.abs(pts[:, AXIS] - 0.5) < 1e-12)]
    model = _solve_sets(mesh, [nodes], c_n, 7.0)
    path = tmp_path / "f.vtu"
    model.write_fracture_vtu(str(path))
    flat = {
        k: np.concatenate([np.asarray(b).reshape(len(b), -1) for b in v])
        for k, v in meshio.read(str(path)).cell_data.items()
    }

    def rebuild(q):
        """Exactly what the Calculator computes."""
        return (
            flat[f"{q}_n"] * flat["normal"]
            + flat[f"{q}_t1"] * flat["tangent1"]
            + flat[f"{q}_t2"] * flat["tangent2"]
        )

    n_faces = model.n_fracture_faces
    want_t = np.tile([0.0, 0.0, -LOAD], (n_faces, 1))
    want_j = np.tile([0.0, 0.0, -LOAD / c_n], (n_faces, 1))
    assert np.allclose(rebuild("traction"), want_t, atol=1e-10)
    assert np.allclose(rebuild("jump"), want_j, atol=1e-10)
    # and the traces rebuild to a difference that is that same opening
    assert np.allclose(rebuild("u_c_plus") - rebuild("u_c_minus"), want_j, atol=1e-10)


def test_each_fracture_is_coherently_oriented():
    r"""One normal direction per fracture, not per facet.

    A facet's canonical normal is a property of that facet alone, so across a
    fracture they come out in no order -- on a simplicial box, alternating +z and
    -z over the eight faces of one plane. The scalars do not care: sigma_nn and
    [[u]].n are invariant under n -> -n. But the frame is what carries them back
    into space, so without orienting the faces the reconstruction alternates
    direction over a flat fracture and every second glyph is reversed.

    THE PLANES HERE CROSS, which is the case that shows it. Their faces share
    edges, so spreading an orientation over shared edges without confining it to
    one fracture compares nearly perpendicular normals, and the sign of that dot
    product is noise.
    """
    mesh, sets = _two_planes(n=4)
    model = _solve_sets(mesh, sets, 50.0, 5.0, SHEAR, labels=[1, 2])
    fr = model.fracture_faces()
    assert len(set(np.unique(fr["id"]))) == 2
    for i in np.unique(fr["id"]):
        sel = fr["id"] == i
        n, a = fr["normal"][sel], fr["area"][sel]
        assert sel.sum() > 4, "too few faces for the orientation to be testable"
        mean = (n * a[:, None]).sum(0)
        mean /= np.linalg.norm(mean)
        # every face agrees with the fracture's mean normal, and these are flat
        # fractures so it agrees exactly
        assert np.allclose(n @ mean, 1.0, atol=1e-12)
        # the frame stays orthonormal and right-handed after any flip
        t1, t2 = fr["tangent1"][sel], fr["tangent2"][sel]
        assert np.allclose(np.cross(t1, t2), n, atol=1e-12)
        assert np.allclose((n * t1).sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose((n * t2).sum(axis=1), 0.0, atol=1e-12)

    # and the law still holds in the reported numbers, flipped faces included
    assert np.allclose(fr["traction"][:, 0], 50.0 * fr["jump"][:, 0], atol=1e-10)
    assert np.allclose(fr["traction"][:, 1:], 5.0 * fr["jump"][:, 1:], atol=1e-10)
    # the jump is still the difference of the traces after the sides were swapped
    assert np.allclose(fr["jump"], fr["u_plus"] - fr["u_minus"], atol=1e-14)


# -- the traction is sigma . n, and the limits of the law ----------------------
#
# THE CHECK THE REST WERE MISSING. Every tangential assertion above is made on an
# axis-aligned fracture under an axial load, where sigma . t1 and sigma . t2are
# zero -- so they confirm zeros and would pass with the tangential slots read
# from the wrong place. An AFFINE datum fixes the stress to a constant tensor
# known before anything is discretized,
#
#     sigma = 2 mu sym(G) + lam tr(G) I ,
#
# and then a stiff fracture must report exactly that tensor contracted with its
# own frame, shear included, on every face however the fracture is inclined.

SIGMA = (2.0 * MU * 0.5 * (np.array(SHEAR).reshape(3, 3) + np.array(SHEAR).reshape(3, 3).T)
         + LAM * np.trace(np.array(SHEAR).reshape(3, 3)) * np.eye(3))


def _affine(mesh, node_sets, c_n, which="weak"):
    """The affine datum on every outer facet, and a fracture inside it."""
    model = mk.CauchyMechanicsModel(mesh, 3, mk.ElasticMaterial(MU, LAM), *FAMILIES[which])
    model.fracture(node_sets, list(range(1, len(node_sets) + 1)))
    if c_n > 0.0:
        model.linear_contact(c_n, c_n)
    else:
        model.bond()
    for f in model.outer_boundary_facets():
        xe = mk.centroid(mesh, 3, int(mk.cofacet_of(mesh, 3, f)))
        constant = [sum(SHEAR[i * 3 + j] * xe[j] for j in range(3)) for i in range(3)]
        model.prescribe_displacement([f], constant, SHEAR)
    model.solve()
    return model


def _sigma_n_in_frame(fr):
    """sigma . n resolved in each face's own {n, t1, t2}."""
    sn = fr["normal"] @ SIGMA.T
    return np.stack(
        [
            np.einsum("ij,ij->i", sn, fr["normal"]),
            np.einsum("ij,ij->i", sn, fr["tangent1"]),
            np.einsum("ij,ij->i", sn, fr["tangent2"]),
        ],
        axis=1,
    )


@pytest.mark.parametrize("which", sorted(FAMILIES))
@pytest.mark.parametrize("family", [mk.Family.simplex, mk.Family.cartesian])
def test_a_bonded_fracture_transmits_the_exact_traction(which, family):
    """t_c = sigma . n in ALL THREE components, not just the normal one."""
    mesh = mk.box([4, 4, 4], 3, family)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    sets = [
        [int(v) for v in np.flatnonzero(np.abs(pts[:, ax] - 0.5) < 1e-12)] for ax in (0, AXIS)
    ]
    fr = _affine(mesh, sets, 0.0, which).fracture_faces()
    assert np.allclose(fr["traction"], _sigma_n_in_frame(fr), atol=1e-10)
    # and the shear it agrees on is not zero, or the last two columns are vacuous
    assert np.abs(_sigma_n_in_frame(fr)[:, 1:]).max() > 1e-2


def test_the_traction_approaches_sigma_n_as_one_over_the_stiffness():
    """A stiff fracture is a bonded one to O(1/c), which is the rate the law
    predicts: [[u]] = sigma_nn / c, so the stress it perturbs is O(1/c) too."""
    mesh = mk.box([4, 4, 4], 3, mk.Family.simplex)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    sets = [[int(v) for v in np.flatnonzero(np.abs(pts[:, AXIS] - 0.5) < 1e-12)]]
    errors = {}
    for c in (1e4, 1e6, 1e8):
        fr = _affine(mesh, sets, c).fracture_faces()
        errors[c] = np.abs(fr["traction"] - _sigma_n_in_frame(fr)).max()
        # the opening IS the traction over the stiffness, exactly
        assert np.allclose(fr["jump"], fr["traction"] / c, atol=1e-12)
    # a hundredfold stiffer is a hundredfold closer, to within a few percent
    for a, b in ((1e4, 1e6), (1e6, 1e8)):
        assert 80.0 < errors[a] / errors[b] < 120.0, errors
    assert errors[1e8] < 1e-6


def test_a_soft_fracture_is_traction_free():
    """The other limit: D_c -> infinity leaves the faces free, so the traction
    vanishes and the opening saturates at what a free surface would give."""
    mesh = mk.box([4, 4, 4], 3, mk.Family.simplex)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    sets = [[int(v) for v in np.flatnonzero(np.abs(pts[:, AXIS] - 0.5) < 1e-12)]]
    soft = _affine(mesh, sets, 1e-4).fracture_faces()
    softer = _affine(mesh, sets, 1e-6).fracture_faces()
    assert np.abs(soft["traction"]).max() < 1e-3
    assert np.abs(softer["traction"]).max() < np.abs(soft["traction"]).max() / 50.0
    # and the opening has stopped growing: it is the free surface's, not the law's
    assert np.allclose(np.abs(soft["jump"][:, 0]).max(),
                       np.abs(softer["jump"][:, 0]).max(), rtol=1e-3)
