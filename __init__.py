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
"""Class registration in blender"""
import bpy
from bpy.utils import register_class, unregister_class
from . import operators, ui


bl_info = {
    "name" : "Show_loads",
    "author" : "Lucas Nascimento",
    "description" : "",
    "blender" : (4, 2, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}
classes = (
    operators.ShowLoadsOperator,
    ui.BIM_PT_structural_actions
    )


def register():
    """Register classes in blender"""
    for cls in classes:
        register_class(cls)

def unregister():
    """Unregister classes in blender"""
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()

