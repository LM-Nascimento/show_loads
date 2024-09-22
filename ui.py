# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2020, 2021 Dion Moult <dion@thinkmoult.com>
#
# This file is part of Bonsai.
#
# Bonsai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Bonsai is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Bonsai.  If not, see <http://www.gnu.org/licenses/>.

import bpy
import bonsai.bim.helper
from bpy.types import Panel, UIList
from bonsai.bim.ifc import IfcStore
from bonsai.bim.helper import draw_attributes, prop_with_search
from .data import (
    StructuralActionsData,
)


class BIM_UL_structural_activities(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if item:
            row = layout.row(align=True)
            row.label(text=item.name)
            row.label(text=item.applied_load_class)


class BIM_PT_show_structural_activities(Panel):
    bl_label = "Show Loads"
    bl_idname = "BIM_PT_show_structural_activities"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "BIM_PT_tab_structural"

    @classmethod
    def poll(cls, context):
        file = IfcStore.get_file()
        return file and hasattr(file, "schema") and file.schema != "IFC2X3"

    def draw(self, context):

        self.props = context.scene.BIMStructuralProperties

        row = self.layout.row(align=True)
        row.operator(
                "bim.show_loads",
                text="Show loads" ,
                icon="FILTER",
            )
