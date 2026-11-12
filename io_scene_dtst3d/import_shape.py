"""Build a Blender scene from a parsed TSShape.

DTS and CDAE both land here. Node quaternions use the conjugate of the
stored value because the engine's matrix builder writes the transpose of
the usual rotation matrix, and both formats share that convention. The
engine's front face is clockwise, so each triangle is reversed for
Blender, whose front face is counter-clockwise.
"""

import bpy
import bmesh
import mathutils

from .tsshape import SequenceFlags, TQuaternion16, TSShape
from .tsmesh import (
    TSDrawPrimitiveType,
    TSMesh,
    TSMeshFlags,
    TSSkinnedMesh,
)


def _triangle_strip(strip):
    triangles = []
    clockwise = False
    for index in range(len(strip) - 2):
        a, b, c = strip[index], strip[index + 1], strip[index + 2]
        triangles.extend((b, a, c) if clockwise else (a, b, c))
        clockwise = not clockwise
    return triangles


def _triangle_fan(fan):
    return [index for corner in range(1, len(fan) - 1) for index in (fan[0], fan[corner], fan[corner + 1])]


def _material(name):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
        material.specular_intensity = 0
        material.use_nodes = True
        material.use_backface_culling = True
    return material


def _quat_to_blender(quat16: TQuaternion16) -> mathutils.Quaternion:
    quat = mathutils.Quaternion((quat16.w, -quat16.x, -quat16.y, -quat16.z))
    quat.normalize()
    return quat


def _node_matrix(node) -> mathutils.Matrix:
    matrix = _quat_to_blender(node.rotation).to_matrix().to_4x4()
    matrix.translation = mathutils.Vector(node.translation)
    return matrix


def _world_matrices(shape: TSShape):
    cache = {}

    def world(index):
        if index in cache:
            return cache[index]
        node = shape.nodes[index]
        local = _node_matrix(node)
        matrix = world(node.parent_index) @ local if node.parent_index >= 0 else local
        cache[index] = matrix
        return matrix

    return [world(index) for index in range(len(shape.nodes))]


def _apply_node_transform(node, obj):
    obj.location = node.translation
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = _quat_to_blender(node.rotation)


def _link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _empty(name, node):
    obj = _link(bpy.data.objects.new(name, None))
    _apply_node_transform(node, obj)
    return obj


def _is_collision(name: str) -> bool:
    return name.startswith(("colmesh", "colbox", "colsphere", "colcapsule"))


def _is_utility_empty(name: str) -> bool:
    if name == "bb_autobillboard" or name.startswith(("bb_", "nulldetail")):
        return True
    for prefix in ("base", "start", "abase", "astart"):
        suffix = name[len(prefix):]
        if name.startswith(prefix) and suffix.isdigit():
            return True
    return False


def _name(shape, index, fallback):
    if 0 <= index < len(shape.names):
        return shape.names[index]
    return fallback


def _frame_slice(mesh: TSMesh):
    total = len(mesh.vertices)
    frames = mesh.num_frames if mesh.num_frames and mesh.num_frames > 1 else 1
    per_frame = mesh.verts_per_frame
    if frames > 1 and per_frame > 0 and total == per_frame * frames:
        return per_frame, frames
    return total, 1


def _build_mesh_object(shape, shape_object, mesh_index, name, merge_verts):
    mesh = shape.meshes[shape_object.start_mesh_index + mesh_index]
    per_frame, frame_count = _frame_slice(mesh)
    positions = mesh.vertices[:per_frame]
    skinned = isinstance(mesh, TSSkinnedMesh)
    # Influences and vertex frames address the original vertices, so those meshes are not welded.
    use_merge = bool(merge_verts) and not skinned and frame_count == 1 and len(mesh.normals) >= per_frame

    blend = bpy.data.meshes.new(f"DTSMesh{shape_object.start_mesh_index + mesh_index}")
    bmesh_data = bmesh.new()
    uv_layer = bmesh_data.loops.layers.uv.new() if len(mesh.tvertices) >= per_frame else None
    uv2_layer = bmesh_data.loops.layers.uv.new("UV2") if len(mesh.t2vertices) >= per_frame else None
    color_layer = bmesh_data.loops.layers.color.new() if len(mesh.colors) >= per_frame else None

    obj = _link(bpy.data.objects.new(name, blend))

    merged_normals = []
    if use_merge:
        remap = {}
        verts = []

        def vertex(index):
            key = (positions[index], mesh.normals[index])
            found = remap.get(key)
            if found is None:
                found = len(verts)
                remap[key] = found
                verts.append(bmesh_data.verts.new(positions[index]))
                merged_normals.append(mesh.normals[index])
            return found
    else:
        verts = [bmesh_data.verts.new(position) for position in positions]

        def vertex(index):
            return index

    slots = {}
    prim_of_face = []
    mat_of_face = []

    def slot_for(primitive):
        key = primitive.material_index
        if key in slots:
            return slots[key]
        if 0 <= key < len(shape.materials):
            material_name = shape.materials[key].name
        else:
            material_name = f"Mat_{key}"
        slots[key] = len(slots)
        obj.data.materials.append(_material(material_name))
        return slots[key]

    def add_face(i0, i1, i2, slot, prim_index):
        try:
            a, b, c = vertex(i0), vertex(i1), vertex(i2)
            face = bmesh_data.faces.new((verts[a], verts[b], verts[c]))
        except (IndexError, ValueError):
            return
        for loop, index in zip(face.loops, (i0, i1, i2)):
            if uv_layer is not None:
                u, v = mesh.tvertices[index]
                loop[uv_layer].uv = (u, 1.0 - v)
            if uv2_layer is not None:
                u, v = mesh.t2vertices[index]
                loop[uv2_layer].uv = (u, 1.0 - v)
            if color_layer is not None:
                loop[color_layer] = mesh.colors[index]
        face.material_index = slot
        face.smooth = True
        prim_of_face.append(prim_index)
        mat_of_face.append(slot)

    for prim_index, primitive in enumerate(mesh.primitives):
        slot = slot_for(primitive)
        raw = mesh.indices[primitive.start:primitive.start + primitive.num_elements]
        if primitive.type == TSDrawPrimitiveType.Triangles:
            triples = raw
        elif primitive.type == TSDrawPrimitiveType.Strip:
            triples = _triangle_strip(raw)
        elif primitive.type == TSDrawPrimitiveType.Fan:
            triples = _triangle_fan(raw)
        else:
            continue
        for offset in range(0, len(triples) - 2, 3):
            add_face(triples[offset + 2], triples[offset + 1], triples[offset], slot, prim_index)

    bmesh_data.normal_update()
    bmesh_data.to_mesh(blend)
    bmesh_data.free()

    if len(mesh.normals) >= per_frame:
        try:
            if use_merge:
                loops = [merged_normals[loop.vertex_index] for loop in blend.loops]
            else:
                loops = [mesh.normals[loop.vertex_index] for loop in blend.loops]
            blend.normals_split_custom_set(loops)
        except (AttributeError, RuntimeError):
            pass

    _write_face_attribute(blend, "ts_prim_index", prim_of_face)
    _write_face_attribute(blend, "ts_mat_index", mat_of_face)

    if frame_count > 1:
        obj.shape_key_add(name="Basis", from_mix=False)
        for frame in range(1, frame_count):
            key = obj.shape_key_add(name=f"frame_{frame}", from_mix=False)
            start = frame * per_frame
            for index in range(per_frame):
                key.data[index].co = mesh.vertices[start + index]
    return obj


def _write_face_attribute(blend, name, values):
    if len(values) != len(blend.polygons):
        return
    try:
        attribute = blend.attributes.get(name) or blend.attributes.new(name, type="INT", domain="FACE")
        for index, value in enumerate(values):
            attribute.data[index].value = int(value)
    except (AttributeError, RuntimeError):
        pass


BONE_LENGTH = 0.05


def _armature(shape: TSShape):
    arm = bpy.data.armatures.new("TS_Armature")
    obj = _link(bpy.data.objects.new("TS_Armature", arm))
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    worlds = _world_matrices(shape)
    bones = []
    for index, node in enumerate(shape.nodes):
        bone = arm.edit_bones.new(_name(shape, node.name_index, f"bone_{index}"))
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, BONE_LENGTH, 0.0)
        # The bone frame is the node frame, so pose channels are node-local deltas.
        bone.matrix = worlds[index]
        bones.append(bone)
    for index, node in enumerate(shape.nodes):
        if 0 <= node.parent_index < len(bones):
            bones[index].parent = bones[node.parent_index]
    bone_names = [bone.name for bone in bones]
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj, bone_names, worlds


def _attach_to_bone(obj, arm, bone_name, world):
    """Parent obj to a bone so it follows the node, keeping `world` as its rest transform."""
    bone = arm.data.bones[bone_name]
    # Bone parenting is relative to the bone tail.
    rest = bone.matrix_local.copy()
    rest.translation = bone.tail_local
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = (arm.matrix_world @ rest).inverted()
    obj.matrix_basis = world


def _node_holder(shape, index, holders):
    """The Empty standing for a node, created with its ancestors on first use."""
    holder = holders.get(index)
    if holder is None:
        node = shape.nodes[index]
        holder = _empty(_name(shape, node.name_index, f"node_{index}"), node)
        if 0 <= node.parent_index < len(shape.nodes):
            holder.parent = _node_holder(shape, node.parent_index, holders)
        holders[index] = holder
    return holder


def _place_in_hierarchy(shape, obj, index, holders):
    node = shape.nodes[index]
    _apply_node_transform(node, obj)
    if 0 <= node.parent_index < len(shape.nodes):
        obj.parent = _node_holder(shape, node.parent_index, holders)
    holders.setdefault(index, obj)


def _depth(shape, index):
    depth = 0
    while 0 <= index < len(shape.nodes) and depth <= len(shape.nodes):
        index = shape.nodes[index].parent_index
        depth += 1
    return depth


def _bind_matrix(mesh: TSSkinnedMesh, worlds):
    """Node world * initial transform: the engine's bone matrix at the default pose.

    It is the identity when the skin was bound to the default pose. Otherwise it is
    a constant that moves the vertices before the bones deform them.
    """
    node_index = mesh.node_index[0]
    if not mesh.initial_transforms or not 0 <= node_index < len(worlds):
        return mathutils.Matrix.Identity(4)
    raw = mesh.initial_transforms[0]
    initial = mathutils.Matrix([raw[0:4], raw[4:8], raw[8:12], raw[12:16]])
    return worlds[node_index] @ initial


def _bind_skin(obj, shape, mesh, arm, bone_names, worlds):
    if not isinstance(mesh, TSSkinnedMesh) or not mesh.node_index:
        return
    bind = _bind_matrix(mesh, worlds)
    if any(abs(bind[row][col] - (row == col)) > 1e-3 for row in range(4) for col in range(4)):
        obj.matrix_basis = bind
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
    names = {}
    for slot, node_index in enumerate(mesh.node_index):
        if 0 <= node_index < len(shape.nodes):
            names[slot] = bone_names[node_index]
    for bone_name in set(names.values()):
        if bone_name not in obj.vertex_groups:
            obj.vertex_groups.new(name=bone_name)
    for vertex, bone, weight in zip(mesh.vertex_index, mesh.bone_index, mesh.weight):
        bone_name = names.get(bone)
        group = obj.vertex_groups.get(bone_name) if bone_name else None
        if group is not None and weight > 0.0:
            group.add([int(vertex)], float(weight), "ADD")


def _key_sequence(shape: TSShape, arm, bone_names, sequence, fps):
    action = bpy.data.actions.new(name=f"TS_{_name(shape, sequence.name_index, 'Seq')}")
    arm.animation_data.action = action
    keys = max(1, sequence.num_keyframes)
    if keys > 1 and sequence.duration > 0.0:
        frames_per_key = sequence.duration * fps / (keys - 1)
    else:
        frames_per_key = fps / 30.0

    for node_index, node in enumerate(shape.nodes):
        rotates = sequence.rotation_matters.test(node_index)
        moves = sequence.translation_matters.test(node_index)
        scales = sequence.scale_matters.test(node_index)
        if not (rotates or moves or scales):
            continue
        pose = arm.pose.bones[bone_names[node_index]]
        pose.rotation_mode = "QUATERNION"
        rest_quat = _quat_to_blender(node.rotation)
        rest_inverse = rest_quat.inverted()
        rest_pos = mathutils.Vector(node.translation)
        rot_slot = sequence.rotation_matters.rank(node_index) if rotates else None
        pos_slot = sequence.translation_matters.rank(node_index) if moves else None
        scale_slot = sequence.scale_matters.rank(node_index) if scales else None
        previous = None
        for key in range(keys):
            frame = 1.0 + key * frames_per_key
            if rotates:
                anim_quat = rest_quat
                sample = sequence.base_rotation + rot_slot * keys + key
                if 0 <= sample < len(shape.anim_node_rotations):
                    anim_quat = _quat_to_blender(shape.anim_node_rotations[sample])
                basis = rest_inverse @ anim_quat
                # Keep neighbouring keys in one hemisphere so interpolation takes the short way.
                if previous is not None and basis.dot(previous) < 0.0:
                    basis.negate()
                previous = basis
                pose.rotation_quaternion = basis
                pose.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if moves:
                anim_pos = rest_pos
                sample = sequence.base_translation + pos_slot * keys + key
                if 0 <= sample < len(shape.anim_node_translations):
                    anim_pos = mathutils.Vector(shape.anim_node_translations[sample])
                pose.location = rest_inverse @ (anim_pos - rest_pos)
                pose.keyframe_insert(data_path="location", frame=frame)
            if scales:
                pose.scale = _sample_scale(shape, sequence, scale_slot, keys, key)
                pose.keyframe_insert(data_path="scale", frame=frame)


def _key_sequences(shape: TSShape, arm, bone_names):
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    edit = bpy.context.preferences.edit
    interpolation = edit.keyframe_new_interpolation_type
    # The engine blends linearly between keys.
    edit.keyframe_new_interpolation_type = "LINEAR"
    arm.animation_data_create()
    try:
        for index, sequence in enumerate(shape.sequences):
            _key_sequence(shape, arm, bone_names, sequence, fps)
            for pose in arm.pose.bones:
                pose.location = (0.0, 0.0, 0.0)
                pose.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pose.scale = (1.0, 1.0, 1.0)
            action = arm.animation_data.action
            arm.animation_data.action = None
            # Tracks overlap in time, so only the first sequence plays until another track is unmuted.
            track = arm.animation_data.nla_tracks.new()
            track.name = action.name
            track.strips.new(action.name, 1, action)
            track.mute = index > 0
    finally:
        edit.keyframe_new_interpolation_type = interpolation


def _sample_scale(shape, sequence, slot, keys, key):
    if slot is None:
        return (1.0, 1.0, 1.0)
    sample = sequence.base_scale + slot * keys + key
    if sequence.flags & SequenceFlags.UniformScale:
        value = shape.anim_node_uniform_scales[sample] if 0 <= sample < len(shape.anim_node_uniform_scales) else 1.0
        return (value, value, value)
    if sequence.flags & SequenceFlags.AlignedScale and 0 <= sample < len(shape.anim_node_aligned_scales):
        return tuple(shape.anim_node_aligned_scales[sample])
    if sequence.flags & SequenceFlags.ArbitraryScale and 0 <= sample < len(shape.anim_node_arbitrary_scale_factors):
        return tuple(shape.anim_node_arbitrary_scale_factors[sample])
    return (1.0, 1.0, 1.0)


def _detail_empties(shape: TSShape):
    if not shape.details:
        return
    parent = bpy.data.objects.get("Details") or _link(bpy.data.objects.new("Details", None))
    object_names = {_name(shape, obj.name_index, "") for obj in shape.objects}
    for detail in shape.details:
        name = _name(shape, detail.name_index, "")
        lowered = name.lower()
        if not name or lowered in object_names or _is_collision(lowered):
            continue
        if lowered != "bb_autobillboard" and not lowered.startswith(("bb_", "nulldetail")):
            continue
        empty = _link(bpy.data.objects.new(name, None))
        empty["detail_size"] = float(detail.size)
        empty["subshape"] = int(detail.sub_shape_num)
        if detail.sub_shape_num < 0 or lowered.startswith("bb_"):
            empty["bb_dimension"] = int(detail.billboard_dimension)
            empty["bb_detail_level"] = int(detail.billboard_detail_level)
            empty["bb_equator_steps"] = int(detail.billboard_equator_steps)
            empty["bb_polar_steps"] = int(detail.billboard_polar_steps)
            empty["bb_polar_angle"] = float(detail.billboard_polar_angle)
            empty["bb_include_poles"] = int(detail.billboard_include_poles)
        empty.parent = parent


def _lod_name(base, mesh, detail):
    prefix = ""
    if mesh.flags & TSMeshFlags.BillboardZAxis:
        prefix = "bbz_"
    elif mesh.flags & TSMeshFlags.Billboard:
        prefix = "bb_"
    suffix = ""
    if detail is not None and detail.size is not None and detail.size >= 10:
        suffix = f"_a{int(round(detail.size))}"
    return prefix + base + suffix


def build_scene(shape: TSShape, merge_verts: bool = True):
    skinned = any(isinstance(mesh, TSSkinnedMesh) for mesh in shape.meshes)
    arm = bone_names = worlds = None
    if (skinned or shape.sequences) and shape.nodes:
        arm, bone_names, worlds = _armature(shape)
    holders = {}
    created = []

    # Parents first, so a node's object is the holder its children attach to.
    order = sorted(range(len(shape.objects)), key=lambda index: _depth(shape, shape.objects[index].node_index))
    for obj_index in order:
        shape_object = shape.objects[obj_index]
        base = _name(shape, shape_object.name_index, f"object_{obj_index}")
        node_index = shape_object.node_index
        has_node = 0 <= node_index < len(shape.nodes)

        if _is_utility_empty(base.lower()) or shape_object.num_meshes <= 0:
            if has_node and arm is None:
                _node_holder(shape, node_index, holders)
            continue

        subshape = shape.get_sub_shape_for_object(obj_index)
        details = {}
        if subshape >= 0:
            for detail in shape.get_sub_shape_details(subshape):
                if detail.sub_shape_num == subshape and detail.object_detail_num >= 0:
                    details[detail.object_detail_num] = detail

        for mesh_index in range(shape_object.num_meshes):
            global_index = shape_object.start_mesh_index + mesh_index
            if global_index >= len(shape.meshes):
                continue
            mesh = shape.meshes[global_index]
            if not isinstance(mesh, TSMesh) or not mesh.vertices:
                continue
            obj = _build_mesh_object(
                shape, shape_object, mesh_index,
                _lod_name(base, mesh, details.get(mesh_index)),
                merge_verts,
            )
            if has_node and not isinstance(mesh, TSSkinnedMesh):
                if arm is not None:
                    _attach_to_bone(obj, arm, bone_names[node_index], worlds[node_index])
                else:
                    _place_in_hierarchy(shape, obj, node_index, holders)
            # ID properties are signed 32-bit. Bit 31 is the billboard flag.
            obj["ts_flags"] = int(mesh.flags) - (0x100000000 if mesh.flags >= 0x80000000 else 0)
            created.append((obj, mesh))

    if arm is not None:
        for obj, mesh in created:
            _bind_skin(obj, shape, mesh, arm, bone_names, worlds)
        if shape.sequences:
            _key_sequences(shape, arm, bone_names)
    _detail_empties(shape)
