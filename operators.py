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
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.util.attribute
import bonsai.bim.helper
import bonsai.core.structural as core
import bonsai.tool as tool
from math import degrees
from mathutils import Vector, Matrix
from bonsai.bim.ifc import IfcStore

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

class LoadDecorator(bpy.types.Operator):
    """Draw decorations to show strucutural actions"""
    bl_idname = "bim.show_3d_loads"
    bl_label = "Show loads in 3D View"

    def modal(self, context, event):
        if event.type == 'ESC':
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
    
        # the arguments we pass the the callback
        coords, indices, coords_2d, load_info, color = get_positions()
        args = (coords, indices, coords_2d, load_info, color)
        # Add the region OpenGL drawing callback
        # draw in view space with 'POST_VIEW' and 'PRE_VIEW'
        self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback,
                                                                args,
                                                                'WINDOW',
                                                                'POST_VIEW'
                                                                )
        context.window_manager.modal_handler_add(self)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'RUNNING_MODAL'}

def location_3d_to_region_2d(coord, context):
    """Convert from 3D space to 2D screen space"""
    region = context.region
    rv3d = context.region_data
    prj = rv3d.perspective_matrix @ Vector((coord[0], coord[1], coord[2], 1.0))
    width_half = region.width / 2.0
    height_half = region.height / 2.0
    return Vector((width_half + width_half * (prj.x / prj.w),
                   height_half + height_half * (prj.y / prj.w),
                   ))
def get_positions():
    """get load positions in 3D View"""
    coords = []
    indices = []
    load_info = []
    coords_2d = []
    color = []

    list_of_curve_members = tool.Ifc.get().by_type("IfcStructuralCurveMember")
    for member in list_of_curve_members:
        activity_list = [getattr(a, 'RelatedStructuralActivity', None) for a in getattr(member, 'AssignedStructuralActivity', None)
                         if getattr(a, 'RelatedStructuralActivity', None).is_a() == 'IfcStructuralCurveAction']
        if len(activity_list) == 0:
            continue
        # member is a structural curve member
        # get Axis attribute from member -> (IFCDIRECTION)
        # get Representation attribute from member -> (IFCPRODUCTDEFINITIONSHAPE)
        # get Representations attribute from Representation -> (IFCTOPOLOGYREPRESENTATION)
        # get Items attribute from Representations -> (IFCEDGE)
        # get EdgeStart attribute from Items -> (IFCVERTEX)
        # get EdgeEnd attribure from Items -> (IFCVERTEX)
        # using blender just get the global coordinates of the first and second vertex in the mesh

        blender_object = IfcStore.get_element(getattr(member, 'GlobalId', None))
        if blender_object.type == 'MESH':
            start_co = blender_object.matrix_world @ blender_object.data.vertices[0].co
            end_co = blender_object.matrix_world @ blender_object.data.vertices[1].co
        x_axis = Vector(end_co-start_co).normalized()
        direction = getattr(member, 'Axis', None)
        #local coordinates
        z_axis = Vector(getattr(direction, 'DirectionRatios', None)).normalized()
        y_axis = z_axis.cross(x_axis).normalized()
        z_axis = x_axis.cross(y_axis).normalized()
        global_to_local = Matrix(((x_axis.x,y_axis.x,z_axis.x,0),
                                  (x_axis.y,y_axis.y,z_axis.y,0),
                                  (x_axis.z,y_axis.z,z_axis.z,0),
                                  (0,0,0,1)))
        forces_list, const, parabola , sinus = get_XYZ_list(activity_list,global_to_local)
        maxforce = 200 #should be the maximum value expected for the loads
        addindex = len(indices)
        counter = 0
        #z direction
        for i in range(len(forces_list)-1):
            current = forces_list[i]
            nextitem = forces_list[i+1]

            if current[1].z != 0 or nextitem[1].z != 0: #if there is load in the z direction
                negative = start_co +current[0]*x_axis-z_axis
                positive = start_co +current[0]*x_axis+z_axis
                coords.append(negative)
                coords_2d.append((current[0],maxforce))
                load_info.append((sinus[0].z,parabola[0].z,current[1].z+const[0].z))
                color.append((0,0,1))

                coords.append(positive)
                coords_2d.append((current[0],-maxforce))
                load_info.append((sinus[0].z,parabola[0].z,current[1].z+const[0].z))
                color.append((0,0,1))

                indices.append((0+counter+addindex,
                                1+counter+addindex,
                                2+counter+addindex))
                indices.append((3+counter+addindex,
                                1+counter+addindex,
                                2+counter+addindex))
                if i == len(forces_list)-2:
                    negative = start_co +nextitem[0]*x_axis-z_axis
                    positive = start_co +nextitem[0]*x_axis+z_axis
                    coords.append(negative)
                    coords_2d.append((nextitem[0],maxforce))
                    load_info.append((sinus[0].z,parabola[0].z,nextitem[1].z+const[0].z))
                    color.append((0,0,1))

                    coords.append(positive)
                    coords_2d.append((nextitem[0],-maxforce))
                    load_info.append((sinus[0].z,parabola[0].z,nextitem[1].z+const[0].z))
                    color.append((0,0,1))

                counter += 2
    return coords, indices, coords_2d, load_info, color


def get_XYZ_list(activity_list,global_to_local):
    loads, const, par, sinus = get_load_list(activity_list,global_to_local)
    unique_list = getuniquepositionlist(loads)
    final_list = []
    for pos in unique_list:
        v1,v2 = gettotalvalues(pos,loads)
        if v1 == v2:
            final_list.append([pos,v1])
        else:
            final_list.append([pos,v1])
            final_list.append([pos,v2])
    if len(final_list)>2:
        del final_list[0]
        del final_list[-1]
    return final_list, const, par , sinus

def getuniquepositionlist(loads):
    unique = []
    for loadinfo in loads:
        for info in loadinfo:
            if info[0] in unique:
                continue
            unique.append(info[0])
    unique.sort()
    return unique

def interp1d(l1,l2, pos):
    fac = (l2[1]-l1[1])/(l2[0]-l1[0])
    v = l1[1] + fac*(pos-l1[0])
    return v

def interpolate(pos,loadinfo,st,end):
    result = Vector((0,0,0))
    for i in range(3):
        l1 = [loadinfo[st][0],loadinfo[st][2][i]]
        l2 = [loadinfo[end][0],loadinfo[end][2][i]]
        result[i] = interp1d(l1,l2, pos)
    return result

def gettotalvalues(pos,loads):
    v1 = Vector((0,0,0))
    v2 = Vector((0,0,0))

    for loadinfo in loads:
        if pos < loadinfo[0][0] or pos > loadinfo[-1][0]:
            continue
        st = 0
        end = len(loadinfo)-1
        while end-st > 0:
            if pos < loadinfo[st][0] or pos > loadinfo[end][0]:
                break
            if loadinfo[st][0] == pos:
                if loadinfo[st][1] in ['start','middle']:
                    v2 += loadinfo[st][2]
                elif loadinfo[st][1] in ['end','middle']:
                    v1 += loadinfo[st][2]

            elif loadinfo[end][0] == pos:
                if loadinfo[end][1] in ['start','middle']:
                    v2 += loadinfo[end][2]
                elif loadinfo[end][1] in ['end','middle']:
                    v1 += loadinfo[end][2]

            elif end-st == 1:
                v1 += interpolate(pos,loadinfo,st,end)
                v2 += interpolate(pos,loadinfo,st,end)
            st += 1
            end -=1
    return v1, v2

def get_load_list(activity_list,global_to_local):
    """get load list in the local coordinate"""
    forcevalues = Vector((0,0,0))
    momentvalues = Vector((0,0,0))
    constvalues = [forcevalues,momentvalues]
    parabolavalues = [forcevalues,momentvalues]        
    sinusvalues = [forcevalues,momentvalues]
    load_values = []
    fac = 1
    lengthunit = [u for u in tool.Ifc.get().by_type("IfcConversionBasedUnit") 
                  if u.UnitType == 'LENGTHUNIT']
    if len(lengthunit):
        string = lengthunit[0].ConversionFactor.ValueComponent.to_string()
        fac = float(string.split('(')[1].split(')')[0])
    
    for activity in activity_list:
        load = getattr(activity, 'AppliedLoad', None)
        global_or_local = getattr(activity, 'GlobalOrLocal', None)
        #values for linear loads
        if load.is_a() == 'IfcStructuralLoadConfiguration':
            locations = getattr(load, 'Locations', None)
            loads = [l for l in getattr(load, 'Values', None) 
                    if l.is_a() == "IfcStructuralLoadLinearForce"
                    ]
            loadinfo = []
            for i,l in enumerate(loads):
                x = 0 if getattr(l, 'LinearForceX', 0) is None else getattr(l, 'LinearForceX', 0)
                y = 0 if getattr(l, 'LinearForceY', 0) is None else getattr(l, 'LinearForceY', 0)
                z = 0 if getattr(l, 'LinearForceZ', 0) is None else getattr(l, 'LinearForceZ', 0)
                forcevalues = Vector((x,y,z))
                x = 0 if getattr(l, 'LinearMomentX', 0) is None else getattr(l, 'LinearMomentX', 0)
                y = 0 if getattr(l, 'LinearMomentY', 0) is None else getattr(l, 'LinearMomentY', 0)
                z = 0 if getattr(l, 'LinearMomentZ', 0) is None else getattr(l, 'LinearMomentZ', 0)
                momentvalues = Vector((x,y,z))
                if i == 0:
                    descr = 'start'
                elif i == len(loads)-1:
                    descr = 'end'
                else:
                    descr = 'middle'
                loadinfo.append([locations[i][0]*fac,descr,forcevalues,momentvalues])
            load_values.append(loadinfo)
        else:
            l = load
            x = 0 if getattr(l, 'LinearForceX', 0) is None else getattr(l, 'LinearForceX', 0)
            y = 0 if getattr(l, 'LinearForceY', 0) is None else getattr(l, 'LinearForceY', 0)
            z = 0 if getattr(l, 'LinearForceZ', 0) is None else getattr(l, 'LinearForceZ', 0)
            forcevalues = Vector((x,y,z))
            x = 0 if getattr(l, 'LinearMomentX', 0) is None else getattr(l, 'LinearMomentX', 0)
            y = 0 if getattr(l, 'LinearMomentY', 0) is None else getattr(l, 'LinearMomentY', 0)
            z = 0 if getattr(l, 'LinearMomentZ', 0) is None else getattr(l, 'LinearMomentZ', 0)
            momentvalues = Vector((x,y,z))
            if 'CONST' == getattr(activity, 'PredefinedType', None):
                constvalues[0] += forcevalues
                constvalues[1] += momentvalues
            elif 'PARABOLA' == getattr(activity, 'PredefinedType', None):
                parabolavalues[0] += forcevalues
                parabolavalues[1] += momentvalues
            elif 'SINUS' == getattr(activity, 'PredefinedType', None):
                sinusvalues[0] += forcevalues
                sinusvalues[1] += momentvalues
        if global_or_local == 'GLOBAL':
            for l in load_values:
                l[2] = global_to_local @ l[2]
                l[3] = global_to_local @ l[3]
            constvalues[0] = global_to_local @constvalues[0]
            constvalues[1] = global_to_local @constvalues[1]
            parabolavalues[0] = global_to_local @ parabolavalues[0]
            parabolavalues[1] = global_to_local @ parabolavalues[1]
            sinusvalues[0] = global_to_local @ sinusvalues[0]
            sinusvalues[1] = global_to_local @ sinusvalues[1]
    return load_values, constvalues,parabolavalues,sinusvalues


def draw_callback(coords, indices, coords_2d, load_info, color):
    """Draw forces, restrictions and results to the screen"""
    #shader info
    vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
    vert_out.smooth('VEC3', "colour")
    vert_out.smooth('VEC3', "forces")
    vert_out.smooth('VEC2', "co")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.push_constant('FLOAT', "spacing")
    shader_info.push_constant('FLOAT', "arrow")

    shader_info.vertex_in(0, 'VEC3', "position")
    shader_info.vertex_in(1, 'VEC3', "color")
    shader_info.vertex_in(2, 'VEC3', "sin_quad_lin_forces")
    shader_info.vertex_in(3, 'VEC2', "uv_coord")
    
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  colour = color;"
        "  gl_Position = viewProjectionMatrix * vec4(position, 1.0f);"
        "  co = uv_coord;"
        "  forces = sin_quad_lin_forces;"
        "  gl_Position = viewProjectionMatrix * vec4(position, 1.0f);"
        "}"
    )


    shader_info.fragment_source(
    "void main()"
    "{"
        "float x = co.x;"
        "float y = co.y;"
        "float abs_y = abs(y);"

        "float a = abs(mod(x,spacing)-0.5*spacing)*5.0*arrow/spacing;"
        "float b = step(a,abs_y)*(step(abs_y,arrow));"
        "float c = step(0.8*spacing,mod(x+0.4*spacing,spacing))*(step(arrow,abs_y));"

        "float sinvalue = forces.x;"
        "float quadraticvalue = forces.y;"
        "float linearvalue = forces.z;"
        "float f = sin(x*3.1416)*sinvalue"
        	  "+(-4.*x*x+4.*x)*quadraticvalue"
        	  "+linearvalue;"
        "float mask = step(0.,y)*step(y,f)+step(y,0.)*step(f,y);"
    
        "float top = step(abs(y-f),0.2*arrow);"
	    "float d = clamp(0.2+top+b+c,0.0,1.0)*mask;"

        "gl_FragColor = vec4(colour,d);"
    "}"
    )

    shader = gpu.shader.create_from_info(shader_info)
    del vert_out
    del shader_info

    #get the locations here
    #coords = [(0, 0, 4), (-1, 0, 5), (2, 0, 4),(1,0,5)]
    #st_info = [(0,0),(0,1),(2,0),(2,1)]
    #indices = ((0,1,2),(1,2,3))

    

    # set open gl configurations
    original_blend = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')
    original_depth_mask = gpu.state.depth_mask_get()
    gpu.state.depth_mask_set(True)
    original_depth_test = gpu.state.depth_test_get()
    gpu.state.depth_test_set('LESS')
    batch = batch_for_shader(shader, 'TRIS', {"position": coords,"color": color, "sin_quad_lin_forces": load_info,"uv_coord": coords_2d},indices=indices)
    
    matrix = bpy.context.region_data.perspective_matrix
    shader.uniform_float("viewProjectionMatrix", matrix)
    shader.uniform_float("spacing", 0.2) #todo make it customizable
    shader.uniform_float("arrow", 20) #todo make it customizable
    batch.draw(shader)

    # restore opengl defaults
    gpu.state.blend_set(original_blend)
    gpu.state.depth_mask_set(original_depth_mask)
    gpu.state.depth_test_set(original_depth_test)
