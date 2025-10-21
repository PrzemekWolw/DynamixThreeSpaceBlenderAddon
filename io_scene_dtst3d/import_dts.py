import bpy, mathutils, bmesh
import time, struct, sys, os, platform, zipfile

from io_scene_dtst3d.tsshape import *
from io_scene_dtst3d.tsmesh import TSMesh, TSSkinnedMesh, TSNullMesh, TSMeshFlags
from io_scene_dtst3d.tsmateriallist import TSMaterial

# ======================================================
# DEP BOOTSTRAP (msgpack + zstandard from local wheels)
# ======================================================
def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

def _ensure_cdae_deps():
    ok_msgpack = _try_import("msgpack")
    ok_zstd    = _try_import("zstandard")
    if ok_msgpack and ok_zstd:
        return

    addon_dir = os.path.dirname(__file__)
    wheels_dir = os.path.join(addon_dir, "third_party", "wheels")
    site_dir   = os.path.join(addon_dir, "third_party", "sitepkgs")
    os.makedirs(site_dir, exist_ok=True)
    if site_dir not in sys.path:
        sys.path.insert(0, site_dir)

    py = sys.version_info
    cp_tag = f"cp{py.major}{py.minor}"
    plat = sys.platform

    def platform_ok(fname: str) -> bool:
        lname = fname.lower()
        if plat.startswith("win"):
            return ("win" in lname) and ("amd64" in lname or "win_amd64" in lname or "arm64" in lname)
        elif plat == "darwin":
            return "macosx" in lname
        else:
            return ("manylinux" in lname) or ("musllinux" in lname) or ("linux" in lname)

    def install_one(dist_prefix: str) -> bool:
        if _try_import(dist_prefix):
            return True
        if not os.path.isdir(wheels_dir):
            return False
        candidates = []
        for fn in os.listdir(wheels_dir):
            if not fn.endswith(".whl"):
                continue
            lower = fn.lower()
            if not (lower.startswith(dist_prefix.replace("_", "-")) or lower.startswith(dist_prefix)):
                continue
            if cp_tag not in lower:
                continue
            if not platform_ok(lower):
                continue
            candidates.append(fn)
        if not candidates:
            return False

        def score(name):
            s = 0
            if "universal2" in name: s += 10
            if "manylinux2014" in name: s += 8
            if "manylinux" in name: s += 5
            if cp_tag in name: s += 4
            return -s

        candidates.sort(key=score)
        wheel_path = os.path.join(wheels_dir, candidates[0])
        try:
            with zipfile.ZipFile(wheel_path, "r") as zf:
                zf.extractall(site_dir)
        except Exception as e:
            print(f"[CDAE bootstrap] Failed to extract {wheel_path}: {e}")
            return False
        return _try_import(dist_prefix)

    if not ok_msgpack:
        ok_msgpack = install_one("msgpack")
    if not ok_zstd:
        ok_zstd = install_one("zstandard")

    if not (ok_msgpack and ok_zstd):
        raise ImportError(
            "Missing dependencies for CDAE importer. "
            "Place msgpack and zstandard wheels into third_party/wheels, "
            "or install them into Blender’s Python."
        )

# ======================================================
# HELPERS
# ======================================================
def triangle_strip_to_list(strip, clockwise):
    triangle_list = []
    for v in range(len(strip) - 2):
        if clockwise:
            triangle_list.extend([strip[v + 1], strip[v], strip[v + 2]])
        else:
            triangle_list.extend([strip[v], strip[v + 1], strip[v + 2]])
        clockwise = not clockwise

    return triangle_list


def create_material(material_name):
    mtl = bpy.data.materials.get(material_name)
    if mtl is None:
        mtl = bpy.data.materials.new(name=material_name)
        mtl.diffuse_color = (1.0, 1.0, 1.0, 1.0)
        mtl.specular_intensity = 0
        mtl.use_nodes = True
        mtl.use_backface_culling = True

    return mtl


def translate_uv(uv):
    return (uv[0], 1.0 - uv[1])


def translate_vert(vert):
    return (vert[0], vert[1], vert[2])

def _detect_version(filepath: str) -> int:
    with open(filepath, "rb") as f:
        data = f.read(4)
    if len(data) != 4:
        raise ValueError("File too small to detect DTS/CDAE version")
    ver_full = struct.unpack("<i", data)[0]
    return ver_full & 0xFF

def _is_collision_name(lname: str) -> bool:
    return lname.startswith(("colmesh", "colbox", "colsphere", "colcapsule"))

def _is_pure_empty_object_name(lname: str) -> bool:
    if (lname.startswith("base") and lname[4:].isdigit()) or (lname.startswith("start") and lname[5:].isdigit()):
        return True
    if (lname.startswith("abase") and lname[5:].isdigit()) or (lname.startswith("astart") and lname[6:].isdigit()):
        return True
    if lname == "bb_autobillboard":
        return True
    if lname.startswith("bb_"):
        return True
    if lname.startswith("nulldetail"):
        return True
    return False

# Quaternion helpers
def _quat16_to_blender_quat(q16: TQuaternion16, is_cdae: bool):
    qf = q16.to_quat_f()
    x = qf.x * (-1.0 if not is_cdae else 1.0)  # DTS legacy: flip X; CDAE: no flip
    return mathutils.Quaternion((qf.w, x, qf.y, qf.z))

def _node_local_matrix(node: ShapeNode, is_cdae: bool) -> mathutils.Matrix:
    q = _quat16_to_blender_quat(node.rotation, is_cdae)
    R = q.to_matrix().to_4x4()
    M = R.copy()
    M.translation = mathutils.Vector(node.translation)
    return M

def _compute_world_mats(shape: TSShape, is_cdae: bool):
    nodes = shape.nodes
    cache = {}

    def get(i):
        if i in cache:
            return cache[i]
        node = nodes[i]
        M_local = _node_local_matrix(node, is_cdae)
        if node.parent_index >= 0:
            M = get(node.parent_index) @ M_local
        else:
            M = M_local
        cache[i] = M
        return M

    return [get(i) for i in range(len(nodes))]

# ======================================================
# IMPORT (shared)
# ======================================================
def apply_node_transform_to_object(shape_node, ob, is_cdae=False):
    ob.location = translate_vert(shape_node.translation)
    rotation_quaternion = _quat16_to_blender_quat(shape_node.rotation, is_cdae=is_cdae)
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = rotation_quaternion

def create_dummy_object_from_shape_object(shape, shape_object, name_override=None):
    scn = bpy.context.scene

    shape_node = shape.nodes[shape_object.node_index]
    base_name = shape.names[shape_object.name_index]
    name = name_override if name_override else base_name
    is_cdae = getattr(shape, "_from_cdae", False)
    ob = bpy.data.objects.new(name, None)
    apply_node_transform_to_object(shape_node, ob, is_cdae=is_cdae)
    scn.collection.objects.link(ob)

    return ob

def _create_detail_empties(shape: TSShape):
    if not hasattr(shape, "details") or not shape.details:
        return
    scn = bpy.context.scene
    parent = bpy.data.objects.get("Details")
    if parent is None:
        parent = bpy.data.objects.new("Details", None)
        scn.collection.objects.link(parent)
    object_name_set = set()
    for obj in shape.objects:
        if 0 <= obj.name_index < len(shape.names):
            object_name_set.add(shape.names[obj.name_index])
    for det in shape.details:
        if not (0 <= det.name_index < len(shape.names)):
            continue
        name = shape.names[det.name_index]
        lname = name.lower()
        if lname in object_name_set:
            continue
        if _is_collision_name(lname):
            continue
        if lname == "bb_autobillboard" or lname.startswith("bb_") or lname.startswith("nulldetail"):
            empty = bpy.data.objects.new(name, None)
            empty["detail_size"] = float(getattr(det, "size", 0.0))
            empty["subshape"] = int(getattr(det, "sub_shape_num", -1))
            if getattr(det, "sub_shape_num", -1) < 0 or lname.startswith("bb_"):
                empty["bb_dimension"] = int(getattr(det, "billboard_dimension", 0))
                empty["bb_detail_level"] = int(getattr(det, "billboard_detail_level", 0))
                empty["bb_equator_steps"] = int(getattr(det, "billboard_equator_steps", 0))
                empty["bb_polar_steps"] = int(getattr(det, "billboard_polar_steps", 0))
                empty["bb_polar_angle"] = float(getattr(det, "billboard_polar_angle", 0.0))
                empty["bb_include_poles"] = int(getattr(det, "billboard_include_poles", 0))
            scn.collection.objects.link(empty)
            empty.parent = parent

def _ensure_action(arm_obj, name):
    act = bpy.data.actions.get(name)
    if not act:
        act = bpy.data.actions.new(name=name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = act
    return act

def _set_key_quat(act, pb, frame, quat):
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = quat
    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def _set_key_loc(act, pb, frame, loc):
    pb.location = loc
    pb.keyframe_insert(data_path="location", frame=frame)

def _set_key_scale(act, pb, frame, sca):
    pb.scale = sca
    pb.keyframe_insert(data_path="scale", frame=frame)

# ======================================================
# UPDATED: key animation as delta relative to node rest (rotation + translation)
# ======================================================
def build_actions_from_sequences(shape: TSShape, arm_obj: bpy.types.Object, fps: float = 30.0):
    if not getattr(shape, "_sequences", None):
        return

    is_cdae = getattr(shape, "_from_cdae", False)

    node_names = [
        shape.names[n.name_index] if 0 <= n.name_index < len(shape.names) else f"bone_{i}"
        for i, n in enumerate(shape.nodes)
    ]

    for seq in shape._sequences:
        seq_name = shape.names[seq.name_index] if 0 <= seq.name_index < len(shape.names) else "Seq"
        act_name = f"TS_{seq_name}"
        act = _ensure_action(arm_obj, act_name)
        for f in list(act.fcurves):
            act.fcurves.remove(f)

        nkeys = max(1, seq.num_keyframes)
        duration = max(0.0, seq.duration)

        def frame_for_key(k: int) -> int:
            if nkeys <= 1:
                return 1
            return int(round(k * (duration / max(1, nkeys - 1)) * fps) + 1)

        rot_vec = getattr(shape, "_anim_node_rotations", [])
        trn_vec = getattr(shape, "_anim_node_translations", [])
        us_vec  = getattr(shape, "_anim_node_uniform_scales", [])
        as_vec  = getattr(shape, "_anim_node_aligned_scales", [])
        arbS_vec = getattr(shape, "_anim_node_arbitrary_scale_factors", [])

        for node_idx, bone_name in enumerate(node_names):
            pb = arm_obj.pose.bones.get(bone_name)
            if not pb:
                continue

            # Rest components (node defaults)
            rest_quat = _quat16_to_blender_quat(shape.nodes[node_idx].rotation, is_cdae)
            rest_tran = shape.nodes[node_idx].translation

            rot_matters = (seq.rotation_matters.values and ((seq.rotation_matters.values[node_idx // 32] >> (node_idx % 32)) & 1))
            trn_matters = (seq.translation_matters.values and ((seq.translation_matters.values[node_idx // 32] >> (node_idx % 32)) & 1))
            scl_matters = (seq.scale_matters.values and ((seq.scale_matters.values[node_idx // 32] >> (node_idx % 32)) & 1))

            rot_num = seq.rotation_matters.rank(node_idx) if rot_matters else None
            trn_num = seq.translation_matters.rank(node_idx) if trn_matters else None
            scl_num = seq.scale_matters.rank(node_idx)      if scl_matters else None

            for k in range(nkeys):
                f = frame_for_key(k)

                # Rotation: delta relative to rest
                if rot_num is not None:
                    idx = seq.base_rotation + rot_num * nkeys + k
                    if 0 <= idx < len(rot_vec):
                        anim_quat = _quat16_to_blender_quat(rot_vec[idx], is_cdae)
                    else:
                        anim_quat = rest_quat
                else:
                    anim_quat = rest_quat
                rot_delta = rest_quat.inverted() @ anim_quat

                # Translation: delta relative to rest
                if trn_num is not None:
                    idx = seq.base_translation + trn_num * nkeys + k
                    if 0 <= idx < len(trn_vec):
                        anim_tran = trn_vec[idx]
                    else:
                        anim_tran = rest_tran
                else:
                    anim_tran = rest_tran
                loc_delta = (anim_tran[0] - rest_tran[0],
                             anim_tran[1] - rest_tran[1],
                             anim_tran[2] - rest_tran[2])

                # Scale (keep as absolute; rarely critical)
                if scl_num is not None:
                    if seq.flags & SequenceFlags.UniformScale and us_vec:
                        idx = seq.base_scale + scl_num * nkeys + k
                        s = (us_vec[idx] if 0 <= idx < len(us_vec) else 1.0)
                        sca = (s, s, s)
                    elif seq.flags & SequenceFlags.AlignedScale and as_vec:
                        idx = seq.base_scale + scl_num * nkeys + k
                        if 0 <= idx < len(as_vec):
                            sx, sy, sz = as_vec[idx]
                            sca = (sx, sy, sz)
                        else:
                            sca = (1.0, 1.0, 1.0)
                    elif seq.flags & SequenceFlags.ArbitraryScale and arbS_vec:
                        idx = seq.base_scale + scl_num * nkeys + k
                        if 0 <= idx < len(arbS_vec):
                            sx, sy, sz = arbS_vec[idx]
                            sca = (sx, sy, sz)
                        else:
                            sca = (1.0, 1.0, 1.0)
                    else:
                        sca = (1.0, 1.0, 1.0)
                else:
                    sca = (1.0, 1.0, 1.0)

                _set_key_loc(act, pb, f, loc_delta)
                _set_key_quat(act, pb, f, rot_delta)
                _set_key_scale(act, pb, f, sca)

        # Keep NLA setup from your working file
        arm_obj.animation_data_create()
        nla = arm_obj.animation_data.nla_tracks.get(act.name)
        if nla is None:
            nla = arm_obj.animation_data.nla_tracks.new()
            nla.name = act.name
        if not any(strip.action == act for strip in nla.strips):
            strip = nla.strips.new(act.name, 1, act)
            strip.frame_end = max(2, round(duration * fps) + 1)

# ======================================================
# Armature creation
# ======================================================
def build_armature_from_shape(shape: TSShape) -> bpy.types.Object:
    scn = bpy.context.scene
    arm_data = bpy.data.armatures.new("TS_Armature")
    arm_obj = bpy.data.objects.new("TS_Armature", arm_data)
    scn.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    is_cdae = getattr(shape, "_from_cdae", False)
    world_mats = _compute_world_mats(shape, is_cdae)
    ebones = {}
    for i, node in enumerate(shape.nodes):
        name = shape.names[node.name_index] if 0 <= node.name_index < len(shape.names) else f"bone_{i}"
        b = arm_data.edit_bones.new(name)
        # set orientation by matrix: head at translation, tail slightly displaced along +Y of bone matrix
        M = world_mats[i]
        b.head = M.translation
        # derive tail as head + (M * (0,0.05,0) - M * (0,0,0))
        dirv = (M.to_3x3() @ mathutils.Vector((0.0, 0.05, 0.0)))
        if dirv.length < 1e-6:
            dirv = mathutils.Vector((0.0, 0.05, 0.0))
        b.tail = b.head + dirv
        ebones[i] = b
    for i, node in enumerate(shape.nodes):
        if node.parent_index >= 0:
            ebones[i].parent = ebones.get(node.parent_index)
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def _apply_skin_to_object(ob: bpy.types.Object, shape: TSShape, ts_mesh, arm_obj: bpy.types.Object):
    if not isinstance(ts_mesh, TSSkinnedMesh) or arm_obj is None:
        return
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm_obj
    bone_slot_to_name = {}
    for bi, node_idx in enumerate(ts_mesh.node_index):
        if 0 <= node_idx < len(shape.nodes):
            ni = shape.nodes[node_idx].name_index
            bone_slot_to_name[bi] = shape.names[ni] if 0 <= ni < len(shape.names) else f"bone_{node_idx}"
    for name in set(bone_slot_to_name.values()):
        if name not in ob.vertex_groups:
            ob.vertex_groups.new(name=name)
    per_vert = {}
    for vi, bi, w in zip(ts_mesh.vertex_index, ts_mesh.bone_index, ts_mesh.weight):
        if w <= 0.0:
            continue
        vname = bone_slot_to_name.get(bi)
        if not vname:
            continue
        per_vert.setdefault(int(vi), []).append((vname, float(w)))
    for vindex, pairs in per_vert.items():
        for vname, w in pairs:
            vg = ob.vertex_groups.get(vname)
            if vg:
                vg.add([vindex], w, 'ADD')

# ======================================================
# Scene composition
# ======================================================
def _create_scene_from_shape(shape: TSShape):
    scn = bpy.context.scene
    is_cdae = getattr(shape, "_from_cdae", False)
    # decide if we need an armature at all
    has_skin = any(isinstance(m, TSSkinnedMesh) for m in shape.meshes)
    has_sequences = bool(getattr(shape, "_sequences", []))
    arm_obj = None
    if has_skin or has_sequences:
        arm_obj = build_armature_from_shape(shape)
    hierarchy = {}
    created = []  # (ob, ts_mesh)
    for obj_index, shape_object in enumerate(shape.objects):
        base_name = shape.names[shape_object.name_index]
        lname = base_name.lower()
        shape_node = shape.nodes[shape_object.node_index]
        parent_obj = None
        if shape_node.parent_index >= 0:
            parent_obj = hierarchy.get(shape_node.parent_index)
        if _is_pure_empty_object_name(lname):
            empty = create_dummy_object_from_shape_object(shape, shape_object, name_override=base_name)
            if parent_obj is not None and empty is not None:
                empty.parent = parent_obj
                empty.matrix_parent_inverse = parent_obj.matrix_world.inverted()
            hierarchy[shape_object.node_index] = empty
            continue
        if shape_object.num_meshes <= 0:
            fallback = create_dummy_object_from_shape_object(shape, shape_object, name_override=base_name)
            if parent_obj is not None and fallback is not None:
                fallback.parent = parent_obj
                fallback.matrix_parent_inverse = parent_obj.matrix_world.inverted()
            hierarchy[shape_object.node_index] = fallback
            continue
        subshape_index = shape.get_sub_shape_for_object(obj_index)
        subshape_details = shape.get_sub_shape_details(subshape_index) if subshape_index >= 0 else []
        detail_by_object_detail = {}
        for det in subshape_details:
            if det.sub_shape_num == subshape_index and det.object_detail_num >= 0:
                detail_by_object_detail[det.object_detail_num] = det
        created_any = False
        for j in range(shape_object.num_meshes):
            global_mesh_idx = shape_object.start_mesh_index + j
            if global_mesh_idx >= len(shape.meshes):
                continue
            mesh = shape.meshes[global_mesh_idx]
            if not isinstance(mesh, (TSMesh, TSSkinnedMesh)) or not getattr(mesh, "vertices", []):
                continue
            prefix = ""
            if isinstance(mesh, TSMesh) and (mesh.flags & TSMeshFlags.BillboardZAxis) != 0:
                prefix = "bbz_"
            elif isinstance(mesh, TSMesh) and (mesh.flags & TSMeshFlags.Billboard) != 0:
                prefix = "bb_"
            det = detail_by_object_detail.get(j)
            suffix = ""
            if det and getattr(det, "size", None) is not None and det.size >= 10:
                suffix = f"_a{int(round(det.size))}"
            name_override = prefix + base_name + suffix
            # For skinned meshes: DO NOT apply node transforms (avoid double-transform).
            ob = create_mesh_object_from_shape_object(shape, shape_object, j, name_override=name_override)
            if ob is not None:
                created_any = True
                # If the mesh is not skinned, parent under hierarchy and apply node transform (done in create_mesh_object).
                # If skinned, we left it at identity; we can still parent for scene hierarchy if desired.
                if parent_obj is not None and not isinstance(mesh, TSSkinnedMesh):
                    ob.parent = parent_obj
                    ob.matrix_parent_inverse = parent_obj.matrix_world.inverted()
                ob["ts_flags"] = int(mesh.flags) if hasattr(mesh, "flags") else 0
                # Only mark hierarchy anchor on non-skinned (skinned shouldn't define scene offset)
                if not isinstance(mesh, TSSkinnedMesh):
                    hierarchy[shape_object.node_index] = ob
                created.append((ob, mesh))
        if not created_any:
            fallback = create_dummy_object_from_shape_object(shape, shape_object, name_override=base_name)
            if parent_obj is not None and fallback is not None:
                fallback.parent = parent_obj
                fallback.matrix_parent_inverse = parent_obj.matrix_world.inverted()
            hierarchy[shape_object.node_index] = fallback
    # After meshes exist: skin binding (only if an armature was created)
    if arm_obj is not None:
        for ob, ts_mesh in created:
            _apply_skin_to_object(ob, shape, ts_mesh, arm_obj)
        # Build actions/NLA only if armature exists
        if has_sequences:
            build_actions_from_sequences(shape, arm_obj, fps=30.0)
    _create_detail_empties(shape)

def create_mesh_object_from_shape_object(shape, shape_object, shape_mesh_index, name_override=None):
    scn = bpy.context.scene
    shape_node = shape.nodes[shape_object.node_index]
    base_name = shape.names[shape_object.name_index]
    shape_object_name = name_override if name_override else base_name
    shape_mesh = shape.meshes[shape_object.start_mesh_index + shape_mesh_index]
    is_cdae = getattr(shape, "_from_cdae", False)

    material_remap = {}
    me = bpy.data.meshes.new('DTSMesh' + str(shape_object.start_mesh_index + shape_mesh_index))

    bm = bmesh.new()
    bm.from_mesh(me)

    uv_layer = None
    uv2_layer = None
    vc_layer = None

    if len(shape_mesh.tvertices) == len(shape_mesh.vertices):
        uv_layer = bm.loops.layers.uv.new()
    if len(shape_mesh.t2vertices) == len(shape_mesh.vertices):
        uv2_layer = bm.loops.layers.uv.new()
    if len(shape_mesh.colors) == len(shape_mesh.vertices):
        vc_layer = bm.loops.layers.color.new()

    ob = bpy.data.objects.new(shape_object_name, me)
    scn.collection.objects.link(ob)

    # Apply node transform ONLY for non-skinned meshes
    if not isinstance(shape_mesh, TSSkinnedMesh):
        apply_node_transform_to_object(shape_node, ob, is_cdae=is_cdae)
    else:
        # Keep skinned mesh at identity (0 loc, 0 rot, 1 scale)
        ob.location = (0.0, 0.0, 0.0)
        ob.rotation_mode = 'QUATERNION'
        ob.rotation_quaternion = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        ob.scale = (1.0, 1.0, 1.0)

    poly_prim_idx = []
    poly_mat_idx = []

    def ensure_material_slot(prim):
        key = prim.material_index
        if key in material_remap:
            return material_remap[key]
        if 0 <= key < len(shape.materials):
            mname = shape.materials[key].name
        else:
            mname = f"Mat_{key}"
        slot = len(material_remap)
        material_remap[key] = slot
        ob.data.materials.append(create_material(mname))
        return slot

    mesh_indices = shape_mesh.indices
    verts = shape_mesh.vertices
    norms = getattr(shape_mesh, "normals", [])

    use_merge = norms and (len(norms) == len(verts))

    if use_merge:
        vert_remap = {}
        remapped_verts = []

        def get_merged_index(vidx: int) -> int:
            key = (verts[vidx], norms[vidx])
            idx = vert_remap.get(key)
            if idx is None:
                idx = len(remapped_verts)
                vert_remap[key] = idx
                remapped_verts.append(bm.verts.new(translate_vert(verts[vidx])))
            return idx

        def add_face_from_indices(i0, i1, i2, mat_slot, prim_i):
            try:
                a = get_merged_index(i0)
                b = get_merged_index(i1)
                c = get_merged_index(i2)
                face = bm.faces.new((remapped_verts[a], remapped_verts[b], remapped_verts[c]))
                if uv_layer is not None:
                    face.loops[0][uv_layer].uv = translate_uv(shape_mesh.tvertices[i0])
                    face.loops[1][uv_layer].uv = translate_uv(shape_mesh.tvertices[i1])
                    face.loops[2][uv_layer].uv = translate_uv(shape_mesh.tvertices[i2])
                if uv2_layer is not None:
                    face.loops[0][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i0])
                    face.loops[1][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i1])
                    face.loops[2][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i2])
                if vc_layer is not None:
                    face.loops[0][vc_layer] = shape_mesh.colors[i0]
                    face.loops[1][vc_layer] = shape_mesh.colors[i1]
                    face.loops[2][vc_layer] = shape_mesh.colors[i2]
                face.material_index = mat_slot
                face.smooth = True
                poly_prim_idx.append(prim_i)
                poly_mat_idx.append(face.material_index)
            except Exception as e:
                print("Face add error:", e)
    else:
        vertices = [bm.verts.new(translate_vert(v)) for v in verts]

        def add_face_from_indices(i0, i1, i2, mat_slot, prim_i):
            try:
                face = bm.faces.new((vertices[i0], vertices[i1], vertices[i2]))
                if uv_layer is not None:
                    face.loops[0][uv_layer].uv = translate_uv(shape_mesh.tvertices[i0])
                    face.loops[1][uv_layer].uv = translate_uv(shape_mesh.tvertices[i1])
                    face.loops[2][uv_layer].uv = translate_uv(shape_mesh.tvertices[i2])
                if uv2_layer is not None:
                    face.loops[0][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i0])
                    face.loops[1][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i1])
                    face.loops[2][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[i2])
                if vc_layer is not None:
                    face.loops[0][vc_layer] = shape_mesh.colors[i0]
                    face.loops[1][vc_layer] = shape_mesh.colors[i1]
                    face.loops[2][vc_layer] = shape_mesh.colors[i2]
                face.smooth = True
                face.material_index = mat_slot
                poly_prim_idx.append(prim_i)
                poly_mat_idx.append(face.material_index)
            except Exception as e:
                print("Face add error:", e)

    for prim_i, prim in enumerate(shape_mesh.primitives):
        mat_slot = ensure_material_slot(prim)

        if prim.type == TSDrawPrimitiveType.Triangles:
            prim_indices = mesh_indices[prim.start:prim.start + prim.num_elements]
            if is_cdae:
                for x in range(0, len(prim_indices), 3):
                    i0 = prim_indices[x + 0]
                    i1 = prim_indices[x + 1]
                    i2 = prim_indices[x + 2]
                    add_face_from_indices(i0, i1, i2, mat_slot, prim_i)
            else:
                for x in range(0, len(prim_indices), 3):
                    i0 = prim_indices[x + 2]
                    i1 = prim_indices[x + 1]
                    i2 = prim_indices[x + 0]
                    add_face_from_indices(i0, i1, i2, mat_slot, prim_i)
        elif prim.type == TSDrawPrimitiveType.Strip:
            strip_indices = mesh_indices[prim.start:prim.start + prim.num_elements]
            prim_indices = triangle_strip_to_list(strip_indices, False)
            if is_cdae:
                for x in range(0, len(prim_indices), 3):
                    i0 = prim_indices[x + 0]
                    i1 = prim_indices[x + 1]
                    i2 = prim_indices[x + 2]
                    add_face_from_indices(i0, i1, i2, mat_slot, prim_i)
            else:
                for x in range(0, len(prim_indices), 3):
                    i0 = prim_indices[x + 2]
                    i1 = prim_indices[x + 1]
                    i2 = prim_indices[x + 0]
                    add_face_from_indices(i0, i1, i2, mat_slot, prim_i)
        else:
            print(f"Unsupported prim type {prim.type}, ignoring.")

    if is_cdae:
        try:
            bmesh.ops.reverse_faces(bm, faces=bm.faces)
        except Exception:
            for f in bm.faces:
                try:
                    f.normal_flip()
                except Exception:
                    pass

    bm.normal_update()
    bm.to_mesh(me)
    bm.free()

    if getattr(shape_mesh, "normals", None) and len(shape_mesh.normals) == len(shape_mesh.vertices):
        try:
            ln = []
            for loop in me.loops:
                vi = loop.vertex_index
                nx, ny, nz = shape_mesh.normals[vi]
                ln.append((nx, ny, nz))
            me.use_auto_smooth = True
            me.normals_split_custom_set(ln)
        except Exception as e:
            print("Failed to set custom split normals:", e)

    try:
        if poly_prim_idx and len(poly_prim_idx) == len(me.polygons):
            attr_p = me.attributes.get("ts_prim_index")
            if attr_p is None:
                attr_p = me.attributes.new("ts_prim_index", type='INT', domain='FACE')
            for i, poly in enumerate(me.polygons):
                attr_p.data[i].value = int(poly_prim_idx[i])
        if poly_mat_idx and len(poly_mat_idx) == len(me.polygons):
            attr_m = me.attributes.get("ts_mat_index")
            if attr_m is None:
                attr_m = me.attributes.new("ts_mat_index", type='INT', domain='FACE')
            for i, poly in enumerate(me.polygons):
                attr_m.data[i].value = int(poly_mat_idx[i])
    except Exception as e:
        print("Failed to write primitive/material face attributes:", e)

    if isinstance(shape_mesh, TSMesh) and not isinstance(shape_mesh, TSSkinnedMesh):
        num_frames = getattr(shape_mesh, "num_frames", 1)
        base_count = len(shape_mesh.vertices)
        if num_frames and num_frames > 1:
            total = len(shape_mesh.vertices)
            if total == base_count * num_frames:
                if not me.shape_keys:
                    ob.shape_key_add(name="Basis", from_mix=False)
                for fi in range(1, num_frames):
                    key = ob.shape_key_add(name=f"frame_{fi}", from_mix=False)
                    start = fi * base_count
                    for vi in range(base_count):
                        key.data[vi].co = mathutils.Vector(translate_vert(shape_mesh.vertices[start + vi]))
            else:
                pass

    return ob

# ======================================================
# DTS import entry
# ======================================================
def read_dts_file(file, filepath):
    shape = TSShape()
    shape.read_from_path(filepath)
    _create_scene_from_shape(shape)

def load_dts(filepath, context):
    print("importing DTS: %r..." % (filepath))
    t0 = time.perf_counter()
    with open(filepath, 'rb'):
        pass
    read_dts_file(None, filepath)
    print(" done in %.4f sec." % (time.perf_counter() - t0))

# ======================================================
# CDAE v31+ support (msgpack + optional zstd)
# ======================================================
def _read_header_v31(f):
    header_size = struct.unpack("<I", f.read(4))[0]
    header_bytes = f.read(header_size)
    import msgpack
    mp = msgpack.unpackb(header_bytes, strict_map_key=False, raw=False)
    is_compressed = bool(mp.get("compression", False))
    bodysize = int(mp.get("bodysize", 0))
    return is_compressed, bodysize

class _UnpackStream:
    def __init__(self, data: bytes):
        import msgpack
        # Some msgpack builds (fallback.py) enforce conservative limits by default.
        # Use very high limits and disable the overall buffer limit.
        kwargs = dict(strict_map_key=False, raw=False)
        caps = dict(
            max_buffer_size=0,           # 0 = unlimited buffer
            max_bin_len=2**31 - 1,
            max_array_len=2**31 - 1,
            max_map_len=2**31 - 1,
            max_str_len=2**31 - 1,
        )
        for k, v in list(caps.items()):
            try:
                msgpack.Unpacker(**kwargs, **{k: v})
                kwargs[k] = v
            except TypeError:
                pass
        self._u = msgpack.Unpacker(**kwargs)
        self._u.feed(data)

    def next(self):
        return self._u.unpack()

def _read_packed_vector(u: "_UnpackStream"):
    size = int(u.next())
    elem_size = int(u.next())
    blob = u.next()
    if not isinstance(blob, (bytes, bytearray)) or len(blob) != size * elem_size:
        raise ValueError("pack_vector size mismatch")
    return size, elem_size, blob

def _unpack_array(blob: bytes, fmt: str):
    es = struct.calcsize("<" + fmt)
    n = len(blob) // es
    out = []
    off = 0
    for _ in range(n):
        out.append(struct.unpack_from("<" + fmt, blob, off))
        off += es
    return out

def _decode_color_bytes_to_rgba_floats(blob: bytes):
    out = []
    for i in range(0, len(blob), 4):
        out.append((blob[i]/255.0, blob[i+1]/255.0, blob[i+2]/255.0, blob[i+3]/255.0))
    return out

def _triangulate_strip(indices, start, count):
    tris = []
    for i in range(count - 2):
        a = indices[start + i + 0]
        b = indices[start + i + 1]
        c = indices[start + i + 2]
        tris.extend([a, b, c] if (i & 1) == 0 else [a, c, b])
    return tris

def _unpack_i32_list(blob: bytes):
    cnt = len(blob) // 4
    return list(struct.unpack("<" + "i"*cnt, blob)) if cnt else []

def _unpack_details(blob: bytes, count: int, elem_size: int):
    out = []
    for i in range(count):
        off = i * elem_size
        if off + elem_size > len(blob):
            break
        nameIndex, subShapeNum, objectDetailNum = struct.unpack_from("<iii", blob, off)
        size = 0.0
        avg = -1.0
        maxe = -1.0
        poly = 0
        bbDim = 0
        bbDL = 0
        bbEq = 0
        bbPo = 0
        bbAng = 0.0
        bbInc = 0
        if elem_size >= 16:
            size = struct.unpack_from("<f", blob, off + 12)[0]
        if elem_size >= 20:
            avg = struct.unpack_from("<f", blob, off + 16)[0]
        if elem_size >= 24:
            maxe = struct.unpack_from("<f", blob, off + 20)[0]
        if elem_size >= 28:
            poly = struct.unpack_from("<i", blob, off + 24)[0]
        if elem_size >= 52:
            bbDim, bbDL = struct.unpack_from("<ii", blob, off + 28)
            bbEq, bbPo = struct.unpack_from("<II", blob, off + 36)
            bbAng = struct.unpack_from("<f", blob, off + 44)[0]
            bbInc = struct.unpack_from("<I", blob, off + 48)[0]
        det = ShapeDetail()
        det.name_index = nameIndex
        det.sub_shape_num = subShapeNum
        det.object_detail_num = objectDetailNum
        det.size = size
        det.average_error = avg
        det.max_error = maxe
        det.poly_count = poly
        det.billboard_dimension = bbDim
        det.billboard_detail_level = bbDL
        det.billboard_equator_steps = bbEq
        det.billboard_polar_steps = bbPo
        det.billboard_polar_angle = bbAng
        det.billboard_include_poles = bbInc
        out.append(det)
    return out

def _read_tsintset(u: "_UnpackStream") -> list[int]:
    obj = u.next()
    if isinstance(obj, (bytes, bytearray)):
        cnt = len(obj) // 4
        return list(struct.unpack("<" + "I" * cnt, obj[:cnt * 4])) if cnt else []
    if isinstance(obj, list):
        if all(isinstance(x, int) for x in obj):
            return [int(x) for x in obj]
        candidates = [sub for sub in obj if isinstance(sub, list) and all(isinstance(x, int) for x in sub)]
        if candidates:
            return [int(x) for x in max(candidates, key=len)]
        return [int(x) for x in obj if isinstance(x, int)]
    if isinstance(obj, dict):
        for k in ("values", "v", "data"):
            v = obj.get(k)
            if isinstance(v, list) and all(isinstance(x, int) for x in v):
                return [int(x) for x in v]
    return []

def _read_cdae_shape(filepath: str) -> TSShape:
    _ensure_cdae_deps()
    import msgpack
    try:
        import zstandard as zstd
    except Exception:
        zstd = None

    with open(filepath, "rb") as f:
        version = struct.unpack("<i", f.read(4))[0] & 0xFF
        if version < 31:
            raise Exception(f"Not a CDAE v31+ file (version={version})")
        is_compressed, bodysize = _read_header_v31(f)
        body = f.read(bodysize) if bodysize > 0 else f.read()
        if is_compressed:
            if not zstd:
                raise RuntimeError("CDAE is compressed; 'zstandard' is unavailable.")
            body = zstd.ZstdDecompressor().decompress(body)

    u = _UnpackStream(body)
    shape = TSShape()
    shape._from_cdae = True

    _ = float(u.next()); _ = int(u.next())
    _ = float(u.next()); _ = float(u.next())
    _ = u.next(); _ = u.next()

    _, _, n_blob = _read_packed_vector(u)
    nodes_raw = _unpack_array(n_blob, "iiiii")
    _, _, o_blob = _read_packed_vector(u)
    objects_raw = _unpack_array(o_blob, "iiiiii")
    _, _, s1_blob = _read_packed_vector(u)
    _, _, s2_blob = _read_packed_vector(u)
    _, _, s3_blob = _read_packed_vector(u)
    _, _, s4_blob = _read_packed_vector(u)
    shape._sub_shape_first_node   = _unpack_i32_list(s1_blob)
    shape._sub_shape_first_object = _unpack_i32_list(s2_blob)
    shape._sub_shape_num_nodes    = _unpack_i32_list(s3_blob)
    shape._sub_shape_num_objects  = _unpack_i32_list(s4_blob)

    _, _, drr_blob = _read_packed_vector(u)
    default_rots = _unpack_array(drr_blob, "hhhh")
    _, _, dtr_blob = _read_packed_vector(u)
    default_trans = _unpack_array(dtr_blob, "fff")

    def _unpack_quat16_blob(blob: bytes):
        es = struct.calcsize("<hhhh")
        n = len(blob) // es
        out = []
        off = 0
        for _ in range(n):
            rx, ry, rz, rw = struct.unpack_from("<hhhh", blob, off)
            out.append(TQuaternion16(rx, ry, rz, rw))
            off += es
        return out

    sz, _, blob = _read_packed_vector(u)
    anim_node_rots = _unpack_quat16_blob(blob) if sz else []
    sz, _, blob = _read_packed_vector(u)
    anim_node_trans = _unpack_array(blob, "fff") if sz else []
    sz, _, blob = _read_packed_vector(u)
    anim_node_uniform = list(struct.unpack("<" + "f"*sz, blob)) if sz else []
    sz, _, blob = _read_packed_vector(u)
    anim_node_aligned = _unpack_array(blob, "fff") if sz else []
    sz, _, blob = _read_packed_vector(u)
    anim_node_arb_factors = _unpack_array(blob, "fff") if sz else []
    sz, _, blob = _read_packed_vector(u)
    anim_node_arb_rots = _unpack_quat16_blob(blob) if sz else []
    _ = _read_packed_vector(u)
    _ = _read_packed_vector(u)

    _obj_sz, _obj_es, _obj_blob = _read_packed_vector(u)
    _trg_sz, _trg_es, _trg_blob = _read_packed_vector(u)

    det_sz, det_es, det_blob = _read_packed_vector(u)
    shape._details = _unpack_details(det_blob, det_sz, det_es) if det_sz else []

    num_names = int(u.next())
    names = [str(u.next()) for _ in range(num_names)]

    num_meshes = int(u.next())
    meshes = []
    for _ in range(num_meshes):
        mtype = int(u.next())
        if mtype == MeshType.NullMeshType:
            meshes.append(TSNullMesh())
            continue

        numFrames = int(u.next())
        numMatFrames = int(u.next())
        parentMesh = int(u.next())

        _ = u.next()
        _ = u.next()
        _ = float(u.next())

        v_sz, _, v_blob = _read_packed_vector(u)
        verts = _unpack_array(v_blob, "fff")
        t_sz, _, t_blob = _read_packed_vector(u)
        tverts = _unpack_array(t_blob, "ff")
        t2_sz, _, t2_blob = _read_packed_vector(u)
        t2verts = _unpack_array(t2_blob, "ff") if t2_sz else []
        c_sz, _, c_blob = _read_packed_vector(u)
        colors = _decode_color_bytes_to_rgba_floats(c_blob) if c_sz else []
        n_sz, _, n_blob = _read_packed_vector(u)
        norms = _unpack_array(n_blob, "fff") if n_sz else []
        _ = _read_packed_vector(u)
        p_sz, _, p_blob = _read_packed_vector(u)
        prims_raw = _unpack_array(p_blob, "iii")
        i_sz, _, i_blob = _read_packed_vector(u)
        indices = list(struct.unpack("<" + "I"*i_sz, i_blob)) if i_sz else []
        tan_sz, _, tan_blob = _read_packed_vector(u)
        tangents = _unpack_array(tan_blob, "fff") if tan_sz else []

        vertsPerFrame = int(u.next())
        flags = int(u.next())

        if mtype == MeshType.SkinMeshType:
            initV_sz, _, _ = _read_packed_vector(u)
            initN_sz, _, _ = _read_packed_vector(u)
            xforms_sz, _, xforms_blob = _read_packed_vector(u)
            vi_sz, _, vi_blob = _read_packed_vector(u)
            bi_sz, _, bi_blob = _read_packed_vector(u)
            w_sz,  _, w_blob  = _read_packed_vector(u)
            ni_sz, _, ni_blob = _read_packed_vector(u)

        tri_indices = []
        tri_prims = []
        for (start, num, mat_idx) in prims_raw:
            ptype = (mat_idx & TSDrawPrimitiveType.TypeMask)
            mindex = (mat_idx & TSDrawPrimitiveType.MaterialMask)
            if ptype == TSDrawPrimitiveType.Triangles:
                out_start = len(tri_indices)
                tri_indices.extend(indices[start:start + num])
                tri_prims.append(TSDrawPrimitive(out_start, len(tri_indices) - out_start, (mindex | TSDrawPrimitiveType.Triangles)))
            elif ptype == TSDrawPrimitiveType.Strip:
                out_start = len(tri_indices)
                tri_indices.extend(_triangulate_strip(indices, start, num))
                tri_prims.append(TSDrawPrimitive(out_start, len(tri_indices) - out_start, (mindex | TSDrawPrimitiveType.Triangles)))
            else:
                out_start = len(tri_indices)
                tri_indices.extend(indices[start:start + num])
                tri_prims.append(TSDrawPrimitive(out_start, len(tri_indices) - out_start, (mindex | TSDrawPrimitiveType.Triangles)))

        if mtype == MeshType.SkinMeshType:
            sm = TSSkinnedMesh()
            sm._vertices  = [tuple(v) for v in verts]
            sm._tvertices = [tuple(t) for t in tverts]
            sm._t2vertices= [tuple(t) for t in t2verts]
            sm._colors    = colors
            sm._indices   = tri_indices
            sm._primitives= tri_prims
            sm._normals   = [tuple(n) for n in norms] if n_sz else []
            sm._tangents  = [tuple(t) for t in tangents] if tan_sz else []
            sm._parent_mesh = parentMesh
            sm._flags = flags
            sm._num_frames = numFrames
            sm._num_mat_frames = numMatFrames
            sm._verts_per_frame = vertsPerFrame
            if xforms_sz:
                xf_floats = struct.unpack("<" + "f" * (xforms_sz * 16), xforms_blob)
                sm._initial_transforms = [list(xf_floats[i * 16:(i + 1) * 16]) for i in range(xforms_sz)]
            sm._vertex_index = list(struct.unpack("<" + "I" * vi_sz, vi_blob)) if vi_sz else []
            sm._bone_index   = list(struct.unpack("<" + "I" * bi_sz, bi_blob)) if bi_sz else []
            sm._weight       = list(struct.unpack("<" + "f" * w_sz,  w_blob))  if w_sz  else []
            sm._node_index   = list(struct.unpack("<" + "I" * ni_sz, ni_blob)) if ni_sz else []
            meshes.append(sm)
        else:
            tm = TSMesh()
            tm._vertices  = [tuple(v) for v in verts]
            tm._tvertices = [tuple(t) for t in tverts]
            tm._t2vertices= [tuple(t) for t in t2verts]
            tm._colors    = colors
            tm._indices   = tri_indices
            tm._primitives= tri_prims
            tm._normals   = [tuple(n) for n in norms] if n_sz else []
            tm._tangents  = [tuple(t) for t in tangents] if tan_sz else []
            tm._parent_mesh = parentMesh
            tm._flags = flags
            tm._num_frames = numFrames
            tm._num_mat_frames = numMatFrames
            tm._verts_per_frame = vertsPerFrame
            meshes.append(tm)

    for i, m in enumerate(meshes):
        if isinstance(m, TSMesh) and m.parent_mesh is not None and m.parent_mesh >= 0:
            p = meshes[m.parent_mesh] if m.parent_mesh < len(meshes) else None
            if isinstance(p, TSMesh) and not m.vertices:
                m.copy_vertex_data_from(p)

    num_seqs = int(u.next())
    shape._sequences = []
    for _ in range(num_seqs):
        seq = ShapeSequence()
        seq.name_index = int(u.next())
        seq.flags = int(u.next())
        seq.num_keyframes = int(u.next())
        seq.duration = float(u.next())
        seq.priority = int(u.next())
        seq.first_ground_frame = int(u.next())
        seq.num_ground_frames = int(u.next())
        seq.base_rotation = int(u.next())
        seq.base_translation = int(u.next())
        seq.base_scale = int(u.next())
        seq.base_object_state = int(u.next())
        seq.base_decal_state = int(u.next())
        seq.first_trigger = int(u.next())
        seq.num_triggers = int(u.next())
        seq.tool_begin = float(u.next())
        seq.rotation_matters.from_list(_read_tsintset(u))
        seq.translation_matters.from_list(_read_tsintset(u))
        seq.scale_matters.from_list(_read_tsintset(u))
        seq.vis_matters.from_list(_read_tsintset(u))
        seq.frame_matters.from_list(_read_tsintset(u))
        seq.mat_frame_matters.from_list(_read_tsintset(u))
        shape._sequences.append(seq)

    # materials
    materials = []
    try:
        mcount = int(u.next())
    except Exception:
        mcount = 0
    for _ in range(mcount):
        mname = str(u.next())
        _ = int(u.next()); _ = int(u.next()); _ = int(u.next()); _ = int(u.next())
        _ = float(u.next()); _ = float(u.next())
        materials.append(TSMaterial(mname))

    # Assign materials into the TSMaterialList (no direct setter -> clear+extend)
    try:
        mat_list = getattr(shape, "_material_list", None)
        mats = getattr(mat_list, "materials", None)
        if isinstance(mats, list):
            mats.clear()
            mats.extend(materials)
        else:
            # Fallback for older shapes
            shape._materials = materials
    except Exception:
        shape._materials = materials

    shape._names = names
    shape._nodes = []
    for i, (nameIdx, parentIdx, _fo, _fc, _ns) in enumerate(nodes_raw):
        n = ShapeNode()
        n.name_index = nameIdx
        n.parent_index = parentIdx
        rx, ry, rz, rw = default_rots[i]
        n.rotation = TQuaternion16(rx, ry, rz, rw)
        tx, ty, tz = default_trans[i]
        n.translation = (tx, ty, tz)
        shape._nodes.append(n)

    shape._objects = []
    for (nameIdx, numMeshes, startMeshIndex, nodeIndex, _ns, _fd) in objects_raw:
        o = ShapeObject()
        o.name_index = nameIdx
        o.num_meshes = numMeshes
        o.start_mesh_index = startMeshIndex
        o.node_index = nodeIndex
        shape._objects.append(o)

    shape._meshes = meshes
    shape._anim_node_rotations = anim_node_rots
    shape._anim_node_translations = [tuple(t) for t in anim_node_trans]
    shape._anim_node_uniform_scales = anim_node_uniform
    shape._anim_node_aligned_scales = [tuple(t) for t in anim_node_aligned]
    shape._anim_node_arbitrary_scale_factors = [tuple(t) for t in anim_node_arb_factors]
    shape._anim_node_arbitrary_scale_rot = anim_node_arb_rots

    return shape

def load_cdae(filepath, context):
    print("importing CDAE: %r..." % (filepath))
    t0 = time.perf_counter()
    shape = _read_cdae_shape(filepath)
    _create_scene_from_shape(shape)
    print(" done in %.4f sec." % (time.perf_counter() - t0))

# ======================================================
# Unified loader
# ======================================================
def load(operator, context, filepath=""):
    try:
        ver = _detect_version(filepath)
    except Exception as e:
        operator.report({'ERROR'}, f"Cannot detect DTS/CDAE version: {e}")
        return {'CANCELLED'}

    if ver == 30:
        operator.report({'ERROR'}, "Version 30 files are not supported.")
        return {'CANCELLED'}

    if ver >= 31:
        load_cdae(filepath, context)
        return {'FINISHED'}
    elif ver >= 19:
        load_dts(filepath, context)
        return {'FINISHED'}
    else:
        operator.report({'ERROR'}, f"Unsupported DTS/CDAE version {ver} (too old).")
        return {'CANCELLED'}
