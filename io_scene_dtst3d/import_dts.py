import bpy, mathutils, bmesh
import time, struct, sys, os, platform, zipfile

from io_scene_dtst3d.tsshape import *

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
            triangle_list.extend([strip[v+1], strip[v], strip[v+2]])
        else:
            triangle_list.extend([strip[v], strip[v+1], strip[v+2]])
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
    # abase00/base00, astart01/start01, nulldetail*, bb_autobillboard, bb_*
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

# ======================================================
# IMPORT (shared)
# ======================================================
def apply_node_transform_to_object(shape_node, ob, is_cdae=False):
    ob.location = translate_vert(shape_node.translation)

    rotation_quaternion = shape_node.rotation.to_quat_f()
    if not is_cdae:
        rotation_quaternion.x = rotation_quaternion.x * -1.0
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion =  mathutils.Quaternion((rotation_quaternion.w,
                                                    rotation_quaternion.x,
                                                    rotation_quaternion.y,
                                                    rotation_quaternion.z))


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

def _pick_best_mesh_index(shape, shape_object):
    start = shape_object.start_mesh_index
    end = start + max(0, shape_object.num_meshes)
    end = min(end, len(shape.meshes))
    best = -1
    best_score = -1
    for idx in range(start, end):
        m = shape.meshes[idx]
        if isinstance(m, TSMesh) and m.vertices:
            score = len(m.vertices)
            if score > best_score:
                best_score = score
                best = idx
    return best if best >= 0 else start

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
    apply_node_transform_to_object(shape_node, ob, is_cdae=is_cdae)
    scn.collection.objects.link(ob)

    mesh_indices = shape_mesh.indices
    vertices = [bm.verts.new(translate_vert(v)) for v in shape_mesh.vertices]

    def add_face(a, b, c, mat_index):
        try:
            face = bm.faces.new((vertices[a], vertices[b], vertices[c]))
            if uv_layer is not None:
                face.loops[0][uv_layer].uv = translate_uv(shape_mesh.tvertices[a])
                face.loops[1][uv_layer].uv = translate_uv(shape_mesh.tvertices[b])
                face.loops[2][uv_layer].uv = translate_uv(shape_mesh.tvertices[c])
            if uv2_layer is not None:
                face.loops[0][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[a])
                face.loops[1][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[b])
                face.loops[2][uv2_layer].uv = translate_uv(shape_mesh.t2vertices[c])
            if vc_layer is not None:
                face.loops[0][vc_layer] = shape_mesh.colors[a]
                face.loops[1][vc_layer] = shape_mesh.colors[b]
                face.loops[2][vc_layer] = shape_mesh.colors[c]
            face.material_index = material_remap[mat_index]
            face.smooth = True
        except Exception as e:
            print("Face add error:", e)

    for prim in shape_mesh.primitives:
        if prim.material_index not in material_remap:
            ts_material = shape.materials[prim.material_index]
            material_remap[prim.material_index] = len(material_remap)
            ob.data.materials.append(create_material(ts_material.name))

        if prim.type == TSDrawPrimitiveType.Triangles:
            prim_indices = mesh_indices[prim.start:prim.start+prim.num_elements]
            if is_cdae:
                for x in range(0, len(prim_indices), 3):
                    a = prim_indices[x + 0]
                    b = prim_indices[x + 1]
                    c = prim_indices[x + 2]
                    add_face(a, b, c, prim.material_index)
            else:
                for x in range(0, len(prim_indices), 3):
                    a = prim_indices[x + 2]
                    b = prim_indices[x + 1]
                    c = prim_indices[x + 0]
                    add_face(a, b, c, prim.material_index)

        elif prim.type == TSDrawPrimitiveType.Strip:
            strip_indices = mesh_indices[prim.start:prim.start+prim.num_elements]
            prim_indices = triangle_strip_to_list(strip_indices, False)
            if is_cdae:
                for x in range(0, len(prim_indices), 3):
                    a = prim_indices[x + 0]
                    b = prim_indices[x + 1]
                    c = prim_indices[x + 2]
                    add_face(a, b, c, prim.material_index)
            else:
                for x in range(0, len(prim_indices), 3):
                    a = prim_indices[x + 2]
                    b = prim_indices[x + 1]
                    c = prim_indices[x + 0]
                    add_face(a, b, c, prim.material_index)
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

    return ob

def _create_detail_empties(shape: TSShape):
    """Create empties for billboard/utility details. Skip collision detail names and duplicates with object names."""
    if not hasattr(shape, "details") or not shape.details:
        return
    scn = bpy.context.scene
    parent = bpy.data.objects.get("Details")
    if parent is None:
        parent = bpy.data.objects.new("Details", None)
        scn.collection.objects.link(parent)
    col_prefixes = ("colmesh", "colbox", "colsphere", "colcapsule")

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

def _create_scene_from_shape(shape: TSShape):
    hierarchy = {}
    scn = bpy.context.scene
    for obj_index, shape_object in enumerate(shape.objects):
        base_name = shape.names[shape_object.name_index]
        lname = base_name.lower()
        print(f"Importing shape object {base_name} with {shape_object.num_meshes} meshes")

        # Pure-empty helper objects: create one empty and skip meshes
        if _is_pure_empty_object_name(lname):
            empty = create_dummy_object_from_shape_object(shape, shape_object, name_override=base_name)
            shape_node = shape.nodes[shape_object.node_index]
            parent = None if shape_node.parent_index < 0 else hierarchy.get(shape_node.parent_index)
            if parent is not None and empty is not None:
                empty.parent = parent
                empty.matrix_parent_inverse = parent.matrix_world.inverted()
            hierarchy[shape_object.node_index] = empty
            continue

        if shape_object.num_meshes <= 0:
            print(f"Not creating object for {base_name}: no assigned mesh")
            continue

        subshape_index = shape.get_sub_shape_for_object(obj_index)
        subshape_details = shape.get_sub_shape_details(subshape_index) if subshape_index >= 0 else []

        # Map objectDetailNum -> detail; _a{size} if size >= 10
        detail_by_object_detail = {}
        for det in subshape_details:
            if det.sub_shape_num == subshape_index and det.object_detail_num >= 0:
                detail_by_object_detail[det.object_detail_num] = det

        shape_node = shape.nodes[shape_object.node_index]
        parent = None if shape_node.parent_index < 0 else hierarchy.get(shape_node.parent_index)

        created_any = False
        for j in range(shape_object.num_meshes):
            global_mesh_idx = shape_object.start_mesh_index + j
            if global_mesh_idx >= len(shape.meshes):
                continue

            mesh = shape.meshes[global_mesh_idx]
            det = detail_by_object_detail.get(j)
            name_suffix = ""
            if det and getattr(det, "size", None) is not None and det.size >= 10:
                name_suffix = f"{int(round(det.size))}"
            name_override = base_name + name_suffix

            if isinstance(mesh, TSMesh) and mesh.vertices:
                # Import collision object geometry too (as real meshes)
                created_object = create_mesh_object_from_shape_object(
                    shape, shape_object, j, name_override=name_override
                )
                if created_object is not None:
                    created_any = True
                    hierarchy[shape_object.node_index] = created_object
                    if parent is not None:
                        created_object.parent = parent
                        created_object.matrix_parent_inverse = parent.matrix_world.inverted()
            elif isinstance(mesh, TSNullMesh):
                # Skip NullMesh slots (no empties for collision objects or others)
                continue

        if not created_any:
            # Fallback: one empty with base name
            created_object = create_dummy_object_from_shape_object(shape, shape_object, name_override=base_name)
            if parent is not None and created_object is not None:
                created_object.parent = parent
                created_object.matrix_parent_inverse = parent.matrix_world.inverted()

    # Detail-level empties for bb/nulldetail (skip collisions and duplicates)
    _create_detail_empties(shape)

# ======================================================
# DTS import entry
# ======================================================
def read_dts_file(file, filepath):
    shape = TSShape()
    shape.read_from_path(filepath)
    _create_scene_from_shape(shape)

def load_dts(filepath, context):
    print("importing DTS: %r..." % (filepath))

    time1 = time.perf_counter()
    file = open(filepath, 'rb')

    read_dts_file(file, filepath)

    print(" done in %.4f sec." % (time.perf_counter() - time1))
    file.close()

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
        self._u = msgpack.Unpacker(strict_map_key=False, raw=False)
        self._u.feed(data)
    def next(self):
        return self._u.unpack()

def _read_packed_vector(u: _UnpackStream):
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

def _unpack_details(blob: bytes, count: int):
    fmt = "<iii f f f i i i I I f I"
    sz = struct.calcsize(fmt)
    out = []
    for i in range(count):
        off = i * sz
        (nameIndex, subShapeNum, objectDetailNum,
         size, avg, maxe, poly,
         bbDim, bbDL, bbEq, bbPo, bbAng, bbInc) = struct.unpack_from(fmt, blob, off)
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

    nsize, _, n_blob = _read_packed_vector(u)
    nodes_raw = _unpack_array(n_blob, "iiiii")
    osize, _, o_blob = _read_packed_vector(u)
    objects_raw = _unpack_array(o_blob, "iiiiii")

    s1_sz, s1_es, s1_blob = _read_packed_vector(u)
    s2_sz, s2_es, s2_blob = _read_packed_vector(u)
    s3_sz, s3_es, s3_blob = _read_packed_vector(u)
    s4_sz, s4_es, s4_blob = _read_packed_vector(u)
    shape._sub_shape_first_node   = _unpack_i32_list(s1_blob)
    shape._sub_shape_first_object = _unpack_i32_list(s2_blob)
    shape._sub_shape_num_nodes    = _unpack_i32_list(s3_blob)
    shape._sub_shape_num_objects  = _unpack_i32_list(s4_blob)

    drr_size, _, drr_blob = _read_packed_vector(u)
    default_rots = _unpack_array(drr_blob, "hhhh")
    dtr_size, _, dtr_blob = _read_packed_vector(u)
    default_trans = _unpack_array(dtr_blob, "fff")

    for _ in range(8):
        _ = _read_packed_vector(u)

    _ = _read_packed_vector(u)  # objectStates
    _ = _read_packed_vector(u)  # triggers
    det_sz, det_es, det_blob = _read_packed_vector(u)  # details
    shape._details = _unpack_details(det_blob, det_sz) if det_sz else []

    num_names = int(u.next())
    names = [str(u.next()) for _ in range(num_names)]

    num_meshes = int(u.next())
    meshes = []
    for _ in range(num_meshes):
        mtype = int(u.next())
        if mtype == MeshType.NullMeshType:
            meshes.append(TSNullMesh())
            continue

        _ = int(u.next()); _ = int(u.next())
        parentMesh = int(u.next())
        _ = u.next(); _ = u.next()
        _ = float(u.next())

        v_sz, _, v_blob = _read_packed_vector(u)
        verts = _unpack_array(v_blob, "fff")
        t_sz, _, t_blob = _read_packed_vector(u)
        tverts = _unpack_array(t_blob, "ff")
        t2_sz, _, t2_blob = _read_packed_vector(u)
        t2verts = _unpack_array(t2_blob, "ff") if t2_sz else []
        c_sz, _, c_blob = _read_packed_vector(u)
        colors = _decode_color_bytes_to_rgba_floats(c_blob) if c_sz else []

        _ = _read_packed_vector(u); _ = _read_packed_vector(u)

        p_sz, _, p_blob = _read_packed_vector(u)
        prims_raw = _unpack_array(p_blob, "iii")
        i_sz, _, i_blob = _read_packed_vector(u)
        indices = list(struct.unpack("<" + "I"*i_sz, i_blob)) if i_sz else []

        _ = _read_packed_vector(u)
        _ = int(u.next()); _ = int(u.next())

        if mtype == MeshType.SkinMeshType:
            _ = _read_packed_vector(u); _ = _read_packed_vector(u)
            _ = _read_packed_vector(u)
            _ = _read_packed_vector(u); _ = _read_packed_vector(u)
            _ = _read_packed_vector(u); _ = _read_packed_vector(u)

        tri_indices = []
        tri_prims = []
        for (start, num, mat_idx) in prims_raw:
            ptype = (mat_idx & TSDrawPrimitiveType.TypeMask)
            mindex = (mat_idx & TSDrawPrimitiveType.MaterialMask)
            if ptype == TSDrawPrimitiveType.Triangles:
                out_start = len(tri_indices)
                tri_indices.extend(indices[start:start+num])
                prim = TSDrawPrimitive(out_start, len(tri_indices)-out_start, (mindex | TSDrawPrimitiveType.Triangles))
                tri_prims.append(prim)
            elif ptype == TSDrawPrimitiveType.Strip:
                out_start = len(tri_indices)
                tri_indices.extend(_triangulate_strip(indices, start, num))
                prim = TSDrawPrimitive(out_start, len(tri_indices)-out_start, (mindex | TSDrawPrimitiveType.Triangles))
                tri_prims.append(prim)
            else:
                out_start = len(tri_indices)
                tri_indices.extend(indices[start:start+num])
                prim = TSDrawPrimitive(out_start, len(tri_indices)-out_start, (mindex | TSDrawPrimitiveType.Triangles))
                tri_prims.append(prim)

        tm = TSMesh()
        tm._vertices  = [tuple(v) for v in verts]
        tm._tvertices = [tuple(t) for t in tverts]
        tm._t2vertices= [tuple(t) for t in t2verts]
        tm._colors    = colors
        tm._indices   = tri_indices  # already flat list of ints
        tm._primitives= tri_prims
        meshes.append(tm)

    num_seqs = int(u.next())
    for _ in range(num_seqs):
        for __ in range(15):
            _ = u.next()

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
    shape._materials = materials
    return shape

def load_cdae(filepath, context):
    print("importing CDAE: %r..." % (filepath))
    t0 = time.perf_counter()
    shape = _read_cdae_shape(filepath)
    _create_scene_from_shape(shape)
    print(" done in %.4f sec." % (time.perf_counter() - t0))

# ======================================================
# Unified loader: detect by content, route accordingly
# ======================================================
def load(operator, context, filepath="", ):
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
