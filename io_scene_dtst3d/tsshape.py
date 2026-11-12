import struct
from typing import List, BinaryIO

from .tsmesh import *
from .tsalloc import *
from .tsmateriallist import *
from .integerset import *

#from tsmesh import *
class MeshType:
    StandardMeshType = 0
    SkinMeshType = 1
    DecalMeshType = 2
    SortedMeshType = 3
    NullMeshType = 4

    TypeMask = 7  # result of 0 | 1 | 2 | 3 | 4

class SequenceFlags:
    UniformScale = 1 << 0
    AlignedScale = 1 << 1
    ArbitraryScale = 1 << 2
    Blend = 1 << 3
    Cyclic = 1 << 4
    MakePath = 1 << 5
    HasTranslucency = 1 << 6
    AnyScale = (1 << 0) | (1 << 1) | (1 << 2)

class TQuaternionF:
    x: float
    y: float
    z: float
    w: float

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class TQuaternion16:
    x: int
    y: int
    z: int
    w: int

    MAX_VALUE = 0x7fff 

    def __init__(self, x, y, z, w):
        self.x : int = x
        self.y = y
        self.z = z
        self.w = w   

    def to_quat_f(self):
        return TQuaternionF(self.x / TQuaternion16.MAX_VALUE,
                            self.y / TQuaternion16.MAX_VALUE,
                            self.z / TQuaternion16.MAX_VALUE,
                            self.w / TQuaternion16.MAX_VALUE)

class ShapeNode:
    def __init__(self):
        self.name_index : int = -1
        self.parent_index : int = -1

        self.translation : tuple[int, int, int] = (0, 0, 0)
        self.rotation : TQuaternion16 = TQuaternion16(0, 0, 0, TQuaternion16.MAX_VALUE)

    def assemble(self, ts_alloc):
        self.name_index = ts_alloc.read32()
        self.parent_index = ts_alloc.read32()

        # runtime computed apparently
        ts_alloc.read32()
        ts_alloc.read32()
        ts_alloc.read32()

class ShapeObject:
    def __init__(self):
        self.name_index : int = -1
        self.num_meshes : int = -1
        self.start_mesh_index : int = -1
        self.node_index : int = -1

    def assemble(self, ts_alloc):
        self.name_index = ts_alloc.read32()
        self.num_meshes = ts_alloc.read32()
        self.start_mesh_index = ts_alloc.read32()
        self.node_index = ts_alloc.read32()

        # runtime computed apparently
        ts_alloc.read32()
        ts_alloc.read32()

class ShapeDetail:
    def __init__(self):
        self.name_index : int = -1
        self.sub_shape_num : int = -1
        self.object_detail_num : int = -1
        self.size : float = 0.0
        self.average_error : float = 0.0
        self.max_error : float = 0.0
        self.poly_count : int = 0

        # billboard settings
        self.billboard_dimension : int = 0
        self.billboard_detail_level : int = 0
        self.billboard_equator_steps : int = 0
        self.billboard_polar_steps : int = 0
        self.billboard_polar_angle : int = 0.0
        self.billboard_include_poles : int = 0

    def assemble(self, ts_alloc, version):
        self.name_index = ts_alloc.read32()
        self.sub_shape_num = ts_alloc.read32()
        self.object_detail_num = ts_alloc.read32()
        self.size = ts_alloc.read_float()
        self.average_error = ts_alloc.read_float()
        self.max_error = ts_alloc.read_float()
        self.poly_count = ts_alloc.read32()

        if version >= 26:
            self.billboard_dimension = ts_alloc.read32()
            self.billboard_detail_level = ts_alloc.read32()
            self.billboard_equator_steps = ts_alloc.read32()
            self.billboard_polar_steps = ts_alloc.read32()
            self.billboard_polar_angle = ts_alloc.read_float()
            self.billboard_include_poles = ts_alloc.read32()

class ShapeSequence:
    def __init__(self):
        self.name_index : int = -1
        self.flags : int = 0
        self.duration : float = 0.0
        self.first_ground_frame : int = -1
        self.num_ground_frames : int = 0
        self.priority : int = 0
        self.base_rotation : int = -1
        self.base_translation : int = -1
        self.base_scale : int = -1
        self.base_object_state : int = -1
        self.base_decal_state : int = -1
        self.first_trigger : int = -1
        self.num_triggers : int = 0
        self.num_keyframes : int = 0
        self.tool_begin : float = 0.0
        
        self.rotation_matters : TSIntegerSet = TSIntegerSet() # set of nodes
        self.translation_matters : TSIntegerSet = TSIntegerSet() # set of nodes
        self.scale_matters : TSIntegerSet = TSIntegerSet() # set of nodes
        self.vis_matters : TSIntegerSet = TSIntegerSet() # set of objects
        self.frame_matters : TSIntegerSet = TSIntegerSet() # set of objects
        self.mat_frame_matters : TSIntegerSet = TSIntegerSet() # set of objects


    def read(self, stream : BinaryIO, version):
        reader = stream

        self.name_index = struct.unpack('<i', reader.read(4))[0]
        if version > 21:
            self.flags = struct.unpack('<L', reader.read(4))[0]
        else:
            self.flags = 0

        self.num_keyframes = struct.unpack('<L', reader.read(4))[0]
        self.duration = struct.unpack('<f', reader.read(4))[0]

        if version < 22:
            # old flags
            if reader.read(1)[0] != 0:
                self.flags |= SequenceFlags.Blend
            if reader.read(1)[0] != 0:
                self.flags |= SequenceFlags.Cyclic
            if reader.read(1)[0] != 0:
                self.flags |= SequenceFlags.MakePath

        self.priority = struct.unpack('<i', reader.read(4))[0]
        self.first_ground_frame = struct.unpack('<i', reader.read(4))[0]
        self.num_ground_frames = struct.unpack('<L', reader.read(4))[0]

        if version > 21:
            self.base_rotation = struct.unpack('<i', reader.read(4))[0]
            self.base_translation = struct.unpack('<i', reader.read(4))[0]
            self.base_scale = struct.unpack('<i', reader.read(4))[0]
            self.base_object_state = struct.unpack('<i', reader.read(4))[0]
            self.base_decal_state = struct.unpack('<i', reader.read(4))[0]
        else:
            self.base_rotation = struct.unpack('<i', reader.read(4))[0]
            self.base_translation = self.base_rotation
            self.base_object_state = struct.unpack('<i', reader.read(4))[0]
            self.base_decal_state = struct.unpack('<i', reader.read(4))[0]

        self.first_trigger = struct.unpack('<i', reader.read(4))[0]
        self.num_triggers = struct.unpack('<L', reader.read(4))[0]
        self.tool_begin = struct.unpack('<f', reader.read(4))[0]

        # membership sets
        self.rotation_matters.read(stream)
        if version < 22:
            self.translation_matters.copy_from(self.rotation_matters)
        else:
            self.translation_matters.read(stream)
            self.scale_matters.read(stream)

        dummy_set = TSIntegerSet()
        dummy_set.read(stream) # DEPRECATED: decals
        dummy_set.read(stream) # DEPRECATED: Ifl materials

        self.vis_matters.read(stream)
        self.frame_matters.read(stream)
        self.mat_frame_matters.read(stream)
        
        

class TSShape:
    def __init__(self):
        self._sequences: List[ShapeSequence] = []
        self._details: List[ShapeDetail] = []
        self._meshes: List[TSMesh] = []
        self._nodes: List[ShapeNode] = []
        self._objects: List[ShapeObject] = []
        self._names: List[str] = []
        self._material_list: TSMaterialList = TSMaterialList()
        self._sub_shape_first_node : List[int] = []
        self._sub_shape_num_nodes : List[int] = []
        self._sub_shape_first_object : List[int] = []
        self._sub_shape_num_objects : List[int] = []
        self._version: int = 0
        self._anim_node_rotations: List[TQuaternion16] = []
        self._anim_node_translations: List[tuple] = []
        self._anim_node_uniform_scales: List[float] = []
        self._anim_node_aligned_scales: List[tuple] = []
        self._anim_node_arbitrary_scale_rot: List[TQuaternion16] = []
        self._anim_node_arbitrary_scale_factors: List[tuple] = []

    @property
    def sequences(self) -> List[ShapeSequence]:
        return self._sequences
    
    @property
    def details(self) -> List[ShapeDetail]:
        return self._details
    
    @property
    def materials(self) -> List[TSMaterial]:
        return self._material_list.materials
    
    @property
    def meshes(self) -> List[TSMesh]:
        return self._meshes
    
    @property
    def nodes(self) -> List[ShapeNode]:
        return self._nodes
    
    @property
    def objects(self) -> List[ShapeObject]:
        return self._objects
    
    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def version(self) -> int:
        return self._version

    @property
    def anim_node_rotations(self):
        return self._anim_node_rotations

    @property
    def anim_node_translations(self):
        return self._anim_node_translations

    @property
    def anim_node_uniform_scales(self):
        return self._anim_node_uniform_scales

    @property
    def anim_node_aligned_scales(self):
        return self._anim_node_aligned_scales

    @property
    def anim_node_arbitrary_scale_factors(self):
        return self._anim_node_arbitrary_scale_factors

    @property
    def anim_node_arbitrary_scale_rot(self):
        return self._anim_node_arbitrary_scale_rot

    def get_sub_shape_for_node(self, node_index) -> int:
        for x in range(len(self._sub_shape_first_node)):
            start = self._sub_shape_first_node[x]
            end = start + self._sub_shape_num_nodes[x]
            if (node_index >= start) and (node_index < end):
                return x
        return -1
    
    def get_sub_shape_for_object(self, object_index) -> int:
        for x in range(len(self._sub_shape_first_object)):
            start = self._sub_shape_first_object[x]
            end = start + self._sub_shape_num_objects[x]
            if (object_index >= start) and (object_index < end):
                return x
        return -1
    
    def get_sub_shape_details(self, sub_shape_index) -> List[ShapeDetail]:
        sub_shape_details = []
        for detail in self._details:
            if detail.sub_shape_num == sub_shape_index or detail.sub_shape_num < 0:
                sub_shape_details.append(detail)
        return sub_shape_details
    
    def read(self, stream: BinaryIO):
        reader = stream

        full_version = struct.unpack('<i', reader.read(4))[0] # version and exporter version packed as two 16-bit values
        version = full_version & 0xFF

        if version < 19:
            raise Exception("This DTS file is too old")
        if version >= 30:
            raise Exception(f"Version {version} is not a binary DTS shape")
        self._version = version

        # Version 29 prefixes the memory buffers with the object names so a
        # loader can list them without unpacking the shape.
        if version >= 29:
            name_count = struct.unpack('<i', reader.read(4))[0]
            for _ in range(name_count):
                name_len = struct.unpack('<I', reader.read(4))[0]
                reader.read(name_len)

        size_mem_buffer = struct.unpack('<i', reader.read(4))[0]
        start_u16 = struct.unpack('<i', reader.read(4))[0]
        start_u8 = struct.unpack('<i', reader.read(4))[0]

        buf = reader.read(size_mem_buffer * 4)
        ts_alloc = TSAlloc(buf, size_mem_buffer, start_u16, start_u8)

        self.assemble(ts_alloc, version)

        # sequences
        num_sequences = struct.unpack('<i', reader.read(4))[0]
        for _ in range(num_sequences):
            sequence = ShapeSequence()
            sequence.read(stream, version)
            self._sequences.append(sequence)

        # materials
        self._material_list.read(stream, version)

        # Versions before 22 index rotations and translations into an array
        # that still has the default node transforms in front of the keys.
        if version < 22:
            for sequence in self._sequences:
                sequence.base_translation -= len(self._nodes)
                sequence.base_rotation -= len(self._nodes)
                sequence.base_scale = 0

        self.resolve_parent_meshes()

    def resolve_parent_meshes(self):
        """Copy vertex and skin data onto meshes that only reference a parent."""
        for mesh in self._meshes:
            if not isinstance(mesh, TSMesh) or mesh.parent_mesh < 0:
                continue
            if not (0 <= mesh.parent_mesh < len(self._meshes)):
                continue
            parent = self._meshes[mesh.parent_mesh]
            if not isinstance(parent, TSMesh):
                continue
            if not mesh.vertices:
                mesh.copy_vertex_data_from(parent)
            if isinstance(mesh, TSSkinnedMesh) and isinstance(parent, TSSkinnedMesh) and not mesh.vertex_index:
                mesh.copy_skin_from(parent)

    def read_from_path(self, path: str):
        with open(path, "rb") as f:
            self.read(f)

    def assemble(self, ts_alloc, version: int):
        num_nodes = ts_alloc.read32()
        num_objects = ts_alloc.read32()
        num_decals = ts_alloc.read32()
        num_sub_shapes = ts_alloc.read32()
        num_ifl_materials = ts_alloc.read32()

        if version < 22:
            num_node_rots = num_node_trans = ts_alloc.read32() - num_nodes
            num_node_uniform_scales = num_node_aligned_scales = num_node_arbitrary_scales = 0
        else:
            num_node_rots = ts_alloc.read32()
            num_node_trans = ts_alloc.read32()
            num_node_uniform_scales = ts_alloc.read32()
            num_node_aligned_scales = ts_alloc.read32()
            num_node_arbitrary_scales = ts_alloc.read32()

        num_ground_frames = ts_alloc.read32() if version > 23 else 0
        num_object_states = ts_alloc.read32()
        num_decal_states = ts_alloc.read32()
        num_triggers = ts_alloc.read32()
        num_details = ts_alloc.read32()
        num_meshes = ts_alloc.read32()

        num_skins = ts_alloc.read32() if version < 23 else 0

        num_names = ts_alloc.read32()
        m_smallest_visible_size = ts_alloc.read_float()
        m_smallest_visible_dl = ts_alloc.read32()

        ts_alloc.check_guard()

        radius = ts_alloc.read_float()
        tube_radius = ts_alloc.read_float()
        center = [ts_alloc.read_float() for _ in range(3)]
        bounds_min = [ts_alloc.read_float() for _ in range(3)]
        bounds_max = [ts_alloc.read_float() for _ in range(3)]

        ts_alloc.check_guard()

        # Node data
        for _ in range(num_nodes):
            node = ShapeNode()
            node.assemble(ts_alloc)
            self._nodes.append(node)
            
        ts_alloc.check_guard()

        # Object data
        for _ in range(num_objects):
            obj = ShapeObject()
            obj.assemble(ts_alloc)
            self._objects.append(obj)

        ts_alloc.check_guard()

        # Deprecated decals
        for _ in range(num_decals):
            for _ in range(5):
                ts_alloc.read32()

        ts_alloc.check_guard()

        # Deprecated IFL materials. Count is numIflMaterials, not numDecals.
        for _ in range(num_ifl_materials):
            for _ in range(5):
                ts_alloc.read32()

        ts_alloc.check_guard()

        # Subshape reading
        for _ in range(num_sub_shapes):
            self._sub_shape_first_node.append(ts_alloc.read32())
        for _ in range(num_sub_shapes):
            self._sub_shape_first_object.append(ts_alloc.read32())
        for _ in range(num_sub_shapes):
            ts_alloc.read32()  # deprecated subShapeFirstDecal
        ts_alloc.check_guard()

        for _ in range(num_sub_shapes):
            self._sub_shape_num_nodes.append(ts_alloc.read32())
        for _ in range(num_sub_shapes):
            self._sub_shape_num_objects.append(ts_alloc.read32())
        for _ in range(num_sub_shapes):
            ts_alloc.read32()  # deprecated subShapeNumDecals
        ts_alloc.check_guard()

        # Default rotations and translations
        for x in range(num_nodes):
            quat = TQuaternion16(ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16())
            self._nodes[x].rotation = quat
                
        ts_alloc.align32()

        for x in range(num_nodes):
            self._nodes[x].translation = (ts_alloc.read_float(), ts_alloc.read_float(), ts_alloc.read_float())

        # Node sequence data. These are the keyframes sequences index into.
        self._anim_node_translations = [
            (ts_alloc.read_float(), ts_alloc.read_float(), ts_alloc.read_float())
            for _ in range(num_node_trans)
        ]
        self._anim_node_rotations = [
            TQuaternion16(ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16())
            for _ in range(num_node_rots)
        ]

        ts_alloc.align32()
        ts_alloc.check_guard()

        if version > 21:
            self._anim_node_uniform_scales = [ts_alloc.read_float() for _ in range(num_node_uniform_scales)]
            self._anim_node_aligned_scales = [
                (ts_alloc.read_float(), ts_alloc.read_float(), ts_alloc.read_float())
                for _ in range(num_node_aligned_scales)
            ]
            self._anim_node_arbitrary_scale_factors = [
                (ts_alloc.read_float(), ts_alloc.read_float(), ts_alloc.read_float())
                for _ in range(num_node_arbitrary_scales)
            ]
            self._anim_node_arbitrary_scale_rot = [
                TQuaternion16(ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16(), ts_alloc.read16())
                for _ in range(num_node_arbitrary_scales)
            ]

            ts_alloc.align32()
            ts_alloc.check_guard()

        if version > 23:
            for _ in range(num_ground_frames):
                for _ in range(3):
                    ts_alloc.read32()
            for _ in range(num_ground_frames):
                for _ in range(4):
                    ts_alloc.read16()
            ts_alloc.align32()
            ts_alloc.check_guard()

        for _ in range(num_object_states):
            for _ in range(3):
                ts_alloc.read32()
        ts_alloc.check_guard()

        for _ in range(num_decal_states):
            ts_alloc.read32()
        ts_alloc.check_guard()

        for _ in range(num_triggers):
            ts_alloc.read32()
            ts_alloc.read32()
        ts_alloc.check_guard()

        for _ in range(num_details):
            detail = ShapeDetail()
            detail.assemble(ts_alloc, version)
            self._details.append(detail)

        ts_alloc.check_guard()

        # Meshes
        for _ in range(num_meshes):
            mesh_type = ts_alloc.read32() & MeshType.TypeMask

            if mesh_type == MeshType.StandardMeshType:
                mesh = TSMesh()
            elif mesh_type == MeshType.SortedMeshType:
                mesh = TSSortedMesh()
            elif mesh_type == MeshType.SkinMeshType:
                mesh = TSSkinnedMesh()
            elif mesh_type == MeshType.DecalMeshType:
                mesh = TSDecalMesh()
            elif mesh_type == MeshType.NullMeshType:
                mesh = TSNullMesh()
            else:
                raise NotImplementedError(f"Can't parse mesh of type {mesh_type}")
            if mesh_type != MeshType.NullMeshType:
                mesh.assemble(ts_alloc, version)
            self._meshes.append(mesh)

        ts_alloc.check_guard()

        # Names
        for _ in range(num_names):
            chars = []
            while True:
                b = ts_alloc.read8()
                if b == 0:
                    break
                chars.append(chr(b))
            self._names.append(''.join(chars))

        ts_alloc.align32()
        ts_alloc.check_guard()

        if version < 23:
            # read skinned meshes from old versions
            ts_alloc.skip32(num_details) # detail first skin
            ts_alloc.skip32(num_details) # detail num skins

            ts_alloc.check_guard()

            for _ in range(num_skins):
                mesh = TSSkinnedMesh()
                mesh.assemble(ts_alloc, version)
                self._meshes.append(mesh)

            ts_alloc.check_guard()
