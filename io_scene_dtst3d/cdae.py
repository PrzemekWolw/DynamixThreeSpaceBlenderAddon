"""Read a BeamNG cache Collada shape (DTS version 31).

The body is one MessagePack stream in the order of TSShape::_read_from_v31.
Vectors are packed as (count, element size, raw bytes). Integer sets are
[usedWordCount, [words...]]. A skin mesh omits its vertex and influence
arrays when it has a parent mesh.
"""

import struct
import sys
from array import array

from .integerset import TSIntegerSet
from .tsmateriallist import TSMaterial
from .tsmesh import TSDrawPrimitive, TSMesh, TSNullMesh, TSSkinnedMesh
from .tsshape import (
    MeshType,
    ShapeDetail,
    ShapeNode,
    ShapeObject,
    ShapeSequence,
    TQuaternion16,
    TSShape,
)

_DETAIL = struct.Struct("<iiifffiiiIIfI")


def _as_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return value


def _as_float(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _numbers(blob: bytes, typecode: str) -> array:
    values = array(typecode)
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _vec3(blob: bytes):
    values = _numbers(blob, "f")
    return list(zip(values[0::3], values[1::3], values[2::3]))


def _vec2(blob: bytes):
    values = _numbers(blob, "f")
    return list(zip(values[0::2], values[1::2]))


def _colors(blob: bytes):
    return [
        (blob[i] / 255.0, blob[i + 1] / 255.0, blob[i + 2] / 255.0, blob[i + 3] / 255.0)
        for i in range(0, len(blob), 4)
    ]


class _Body:
    def __init__(self, data: bytes):
        import msgpack
        options = {"strict_map_key": False, "raw": False}
        limits = {
            "max_buffer_size": 2**31 - 1,
            "max_bin_len": 2**31 - 1,
            "max_array_len": 2**31 - 1,
            "max_map_len": 2**31 - 1,
            "max_str_len": 2**31 - 1,
        }
        for key, value in limits.items():
            try:
                msgpack.Unpacker(**options, **{key: value})
            except TypeError:
                continue
            options[key] = value
        self._unpacker = msgpack.Unpacker(**options)
        self._unpacker.feed(data)
        self._out_of_data = msgpack.OutOfData

    def next(self):
        return self._unpacker.unpack()

    def finished(self) -> bool:
        try:
            extra = self.next()
        except self._out_of_data:
            return True
        raise ValueError(f"CDAE body has trailing data starting with {type(extra).__name__}")

    def i32(self) -> int:
        return _as_int(self.next())

    def f32(self) -> float:
        return _as_float(self.next())

    def point(self):
        value = self.next()
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("expected a Point3F array of 3 numbers")
        return tuple(_as_float(item) for item in value)

    def box(self):
        value = self.next()
        if not isinstance(value, list) or len(value) != 6:
            raise ValueError("expected a Box3F array of 6 numbers")
        return tuple(_as_float(item) for item in value)

    def packed(self):
        count = self.i32()
        elem_size = self.i32()
        blob = self.next()
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError("packed vector body is not binary")
        if len(blob) != count * elem_size:
            raise ValueError(f"packed vector is {len(blob)} bytes, expected {count * elem_size}")
        return count, elem_size, bytes(blob)

    def intset(self) -> TSIntegerSet:
        # tsIntegerSet_msgpack.h packs [usedSize, [dword, ...]].
        value = self.next()
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
            raise ValueError("TSIntegerSet must be [usedSize, [words]]")
        used = _as_int(value[0])
        words = value[1]
        if len(words) != used:
            raise ValueError(f"TSIntegerSet declared {used} words but carried {len(words)}")
        result = TSIntegerSet()
        result.from_list(_as_int(word) for word in words)
        return result


def _details(blob: bytes, count: int, elem_size: int):
    details = []
    for index in range(count):
        chunk = blob[index * elem_size:(index + 1) * elem_size]
        if len(chunk) < _DETAIL.size:
            break
        (
            name_index, sub_shape, object_detail,
            size, average_error, max_error, poly_count,
            bb_dimension, bb_detail, bb_equator, bb_polar, bb_angle, bb_poles,
        ) = _DETAIL.unpack_from(chunk)
        detail = ShapeDetail()
        detail.name_index = name_index
        detail.sub_shape_num = sub_shape
        detail.object_detail_num = object_detail
        detail.size = size
        detail.average_error = average_error
        detail.max_error = max_error
        detail.poly_count = poly_count
        detail.billboard_dimension = bb_dimension
        detail.billboard_detail_level = bb_detail
        detail.billboard_equator_steps = bb_equator
        detail.billboard_polar_steps = bb_polar
        detail.billboard_polar_angle = bb_angle
        detail.billboard_include_poles = bb_poles
        details.append(detail)
    return details


def _quats(blob: bytes):
    values = _numbers(blob, "h")
    return [
        TQuaternion16(values[i], values[i + 1], values[i + 2], values[i + 3])
        for i in range(0, len(values), 4)
    ]


def _i32_list(blob: bytes):
    return list(_numbers(blob, "i"))


def _apply_mesh(mesh, decoded):
    mesh._vertices = decoded["vertices"]
    mesh._tvertices = decoded["tvertices"]
    mesh._t2vertices = decoded["t2vertices"]
    mesh._colors = decoded["colors"]
    mesh._normals = decoded["normals"]
    mesh._tangents = decoded["tangents"]
    mesh._primitives = decoded["primitives"]
    mesh._indices = decoded["indices"]
    mesh._parent_mesh = decoded["parent"]
    mesh._flags = decoded["flags"]
    mesh._num_frames = decoded["frames"]
    mesh._num_mat_frames = decoded["mat_frames"]
    mesh._verts_per_frame = decoded["verts_per_frame"]


def _read_mesh(body: _Body):
    mesh_type = body.i32()
    if mesh_type == MeshType.NullMeshType:
        return TSNullMesh()
    if mesh_type not in (MeshType.StandardMeshType, MeshType.SkinMeshType):
        raise ValueError(f"unsupported CDAE mesh type {mesh_type}")

    num_frames = body.i32()
    num_mat_frames = body.i32()
    parent_mesh = body.i32()
    body.box()
    body.point()
    body.f32()

    _, _, verts_blob = body.packed()
    _, _, tverts_blob = body.packed()
    t2_count, _, t2_blob = body.packed()
    color_count, _, color_blob = body.packed()
    normal_count, _, normal_blob = body.packed()
    body.packed()  # encoded normals, deprecated
    _, _, prim_blob = body.packed()
    index_count, _, index_blob = body.packed()
    tangent_count, _, tangent_blob = body.packed()
    verts_per_frame = body.i32()
    flags = body.i32() & 0xFFFFFFFF

    prim_words = _i32_list(prim_blob)
    primitives = [
        TSDrawPrimitive(prim_words[i], prim_words[i + 1], prim_words[i + 2] & 0xFFFFFFFF)
        for i in range(0, len(prim_words), 3)
    ]
    tangents = []
    if tangent_count:
        raw = _numbers(tangent_blob, "f")
        tangents = list(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))

    decoded = {
        "vertices": _vec3(verts_blob),
        "tvertices": _vec2(tverts_blob),
        "t2vertices": _vec2(t2_blob) if t2_count else [],
        "colors": _colors(color_blob) if color_count else [],
        "normals": _vec3(normal_blob) if normal_count else [],
        "tangents": tangents,
        "primitives": primitives,
        "indices": list(_numbers(index_blob, "I")) if index_count else [],
        "parent": parent_mesh,
        "flags": flags,
        "frames": num_frames,
        "mat_frames": num_mat_frames,
        "verts_per_frame": verts_per_frame,
    }

    if mesh_type == MeshType.SkinMeshType:
        mesh = TSSkinnedMesh()
        _apply_mesh(mesh, decoded)
        # initialVerts and initialNorms, then the influence arrays, are only
        # written when this mesh has no parent. initialTransforms always is.
        if parent_mesh < 0:
            body.packed()
            body.packed()
        xform_count, _, xform_blob = body.packed()
        if xform_count:
            floats = _numbers(xform_blob, "f")
            mesh._initial_transforms = [list(floats[i * 16:(i + 1) * 16]) for i in range(xform_count)]
        if parent_mesh < 0:
            vi_count, _, vi_blob = body.packed()
            bi_count, _, bi_blob = body.packed()
            w_count, _, w_blob = body.packed()
            ni_count, _, ni_blob = body.packed()
            mesh._vertex_index = list(_numbers(vi_blob, "I")) if vi_count else []
            mesh._bone_index = list(_numbers(bi_blob, "I")) if bi_count else []
            mesh._weight = list(_numbers(w_blob, "f")) if w_count else []
            mesh._node_index = list(_numbers(ni_blob, "I")) if ni_count else []
        return mesh

    mesh = TSMesh()
    _apply_mesh(mesh, decoded)
    return mesh


def _read_header(handle):
    header_size = struct.unpack("<I", handle.read(4))[0]
    import msgpack
    header = msgpack.unpackb(handle.read(header_size), strict_map_key=False, raw=False)
    if not isinstance(header, dict):
        raise ValueError("CDAE header is not a map")
    return bool(header.get("compression", False)), int(header.get("bodysize", 0))


def read_shape(path: str) -> TSShape:
    with open(path, "rb") as handle:
        version = struct.unpack("<i", handle.read(4))[0] & 0xFF
        if version < 31:
            raise ValueError(f"not a CDAE v31 file (version {version})")
        compressed, body_size = _read_header(handle)
        body = handle.read(body_size) if body_size else handle.read()
        if len(body) != body_size and body_size:
            raise ValueError(f"CDAE body is {len(body)} bytes, header says {body_size}")

    if compressed:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "This CDAE file is Zstandard-compressed. Run tools/fetch_wheels.py "
                "and install the extension so zstandard is available."
            ) from exc
        body = zstd.ZstdDecompressor().decompress(body)

    src = _Body(body)
    shape = TSShape()
    shape._version = version

    src.f32()   # smallest visible size
    src.i32()   # smallest visible detail
    src.f32()   # radius
    src.f32()   # tube radius
    src.point()
    src.box()

    _, _, node_blob = src.packed()
    nodes = _numbers(node_blob, "i")
    _, _, object_blob = src.packed()
    objects = _numbers(object_blob, "i")
    _, _, first_node_blob = src.packed()
    _, _, first_object_blob = src.packed()
    _, _, num_node_blob = src.packed()
    _, _, num_object_blob = src.packed()
    shape._sub_shape_first_node = _i32_list(first_node_blob)
    shape._sub_shape_first_object = _i32_list(first_object_blob)
    shape._sub_shape_num_nodes = _i32_list(num_node_blob)
    shape._sub_shape_num_objects = _i32_list(num_object_blob)

    _, _, rot_blob = src.packed()
    default_rots = _quats(rot_blob)
    _, _, trans_blob = src.packed()
    default_trans = _vec3(trans_blob)

    rot_count, _, anim_rot_blob = src.packed()
    trans_count, _, anim_trans_blob = src.packed()
    scale_count, _, uniform_blob = src.packed()
    aligned_count, _, aligned_blob = src.packed()
    factor_count, _, factor_blob = src.packed()
    arb_count, _, arb_rot_blob = src.packed()
    shape._anim_node_rotations = _quats(anim_rot_blob) if rot_count else []
    shape._anim_node_translations = _vec3(anim_trans_blob) if trans_count else []
    shape._anim_node_uniform_scales = list(_numbers(uniform_blob, "f")) if scale_count else []
    shape._anim_node_aligned_scales = _vec3(aligned_blob) if aligned_count else []
    shape._anim_node_arbitrary_scale_factors = _vec3(factor_blob) if factor_count else []
    shape._anim_node_arbitrary_scale_rot = _quats(arb_rot_blob) if arb_count else []

    # groundTranslations, groundRotations, objectStates, triggers
    for _ in range(4):
        src.packed()

    detail_count, detail_size, detail_blob = src.packed()
    shape._details = _details(detail_blob, detail_count, detail_size) if detail_count else []

    name_count = src.i32()
    shape._names = [str(src.next()) for _ in range(name_count)]

    mesh_count = src.i32()
    shape._meshes = [_read_mesh(src) for _ in range(mesh_count)]
    shape.resolve_parent_meshes()

    sequence_count = src.i32()
    for _ in range(sequence_count):
        sequence = ShapeSequence()
        sequence.name_index = src.i32()
        sequence.flags = src.i32() & 0xFFFFFFFF
        sequence.num_keyframes = src.i32()
        sequence.duration = src.f32()
        sequence.priority = src.i32()
        sequence.first_ground_frame = src.i32()
        sequence.num_ground_frames = src.i32()
        sequence.base_rotation = src.i32()
        sequence.base_translation = src.i32()
        sequence.base_scale = src.i32()
        sequence.base_object_state = src.i32()
        sequence.base_decal_state = src.i32()
        sequence.first_trigger = src.i32()
        sequence.num_triggers = src.i32()
        sequence.tool_begin = src.f32()
        sequence.rotation_matters = src.intset()
        sequence.translation_matters = src.intset()
        sequence.scale_matters = src.intset()
        sequence.vis_matters = src.intset()
        sequence.frame_matters = src.intset()
        sequence.mat_frame_matters = src.intset()
        shape.sequences.append(sequence)

    material_count = src.i32()
    for _ in range(material_count):
        material = TSMaterial(str(src.next()))
        material.flags = src.i32() & 0xFFFFFFFF
        src.i32()   # reflectance map
        src.i32()   # bump map
        src.i32()   # detail map
        src.f32()   # detail scale
        src.f32()   # reflection amount
        shape.materials.append(material)

    node_count = len(nodes) // 5
    if node_count != len(default_rots) or node_count != len(default_trans):
        raise ValueError("CDAE node and default-transform counts differ")
    for index in range(node_count):
        node = ShapeNode()
        base = index * 5
        node.name_index = nodes[base]
        node.parent_index = nodes[base + 1]
        node.rotation = default_rots[index]
        node.translation = default_trans[index]
        shape.nodes.append(node)

    for index in range(0, len(objects), 6):
        obj = ShapeObject()
        obj.name_index = objects[index]
        obj.num_meshes = objects[index + 1]
        obj.start_mesh_index = objects[index + 2]
        obj.node_index = objects[index + 3]
        shape.objects.append(obj)

    src.finished()
    return shape
