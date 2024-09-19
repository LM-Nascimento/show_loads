# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2021 Dion Moult <dion@thinkmoult.com>
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
import bonsai.tool as tool
import ifcopenshell.util.doc


def refresh():
    StructuralActionsData.is_loaded = False

class StructuralActionsData:
    data = {}
    is_loaded = False

    @classmethod
    def load(cls):
        cls.data = {
            "total_actions": cls.total_actions(),
            "actions_classes": cls.actions_classes(),
            "structural_actions_types": cls.structural_actions_types(),
        }
        cls.is_loaded = True

    @classmethod
    def total_actions(cls):
        return len(tool.Ifc.get().by_type("IfcStructuralAction"))

    @classmethod
    def actions_classes(cls):
        return {l.id(): l.is_a() for l in tool.Ifc.get().by_type("IfcStructuralAction")}

    @classmethod
    def structural_actions_types(cls):
        declaration = tool.Ifc.schema().declaration_by_name("IfcStructuralAction")
        version = tool.Ifc.get_schema()
        return [
            (d.name(), d.name(), ifcopenshell.util.doc.get_entity_doc(version, d.name()).get("description", ""))
            for d in declaration.subtypes()
        ]