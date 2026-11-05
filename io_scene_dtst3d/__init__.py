try:
    import bpy
    from bpy.props import BoolProperty, StringProperty
    from bpy_extras.io_utils import ImportHelper
except ImportError:  # so the readers can be checked without Blender
    bpy = None


if bpy is not None:
    class ImportDTS(bpy.types.Operator, ImportHelper):
        """Import a Torque3D DTS shape or a BeamNG cache Collada shape"""
        bl_idname = "import_scene.dtst3d"
        bl_label = "Import Dynamix Three Space / CDAE"
        bl_options = {"UNDO"}

        filename_ext = ".dts;.cdae"
        filter_glob: StringProperty(default="*.dts;*.cdae", options={"HIDDEN"})

        merge_verts: BoolProperty(
            name="Merge Vertices",
            description="Weld vertices that share a position and a normal. Skinned meshes are left unwelded so bone weights keep their vertex indices",
            default=True,
        )

        def execute(self, context):
            from . import import_dts
            keywords = self.as_keywords(ignore=(
                "axis_forward",
                "axis_up",
                "filter_glob",
                "check_existing",
            ))
            return import_dts.load(self, context, **keywords)

    def menu_func_import(self, context):
        del context
        self.layout.separator()
        self.layout.operator(ImportDTS.bl_idname, text="Dynamix Three Space / Cache Collada (.dts, .cdae)")
        self.layout.separator()

    def register():
        bpy.utils.register_class(ImportDTS)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
        bpy.utils.unregister_class(ImportDTS)

    if __name__ == "__main__":
        register()
