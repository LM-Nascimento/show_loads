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
import numpy as np

class ShowLoadsOperator(bpy.types.Operator):
	"""Draw decorations to show strucutural actions in 3d view"""
	bl_idname = "bim.show_loads"
	bl_label = "Show loads in 3D View"

	def modal(self, context, event):
		if event.type == 'ESC':
			LoadsDecorator.uninstall()
			for area in context.screen.areas:
				if area.type == 'VIEW_3D':
					area.tag_redraw()
			return {'FINISHED'}
		return {'PASS_THROUGH'}

	def invoke(self, context, event):
		LoadsDecorator.install()
		LoadsDecorator.update()
		context.window_manager.modal_handler_add(self)
		for area in context.screen.areas:
			if area.type == 'VIEW_3D':
				area.tag_redraw()

		return {'RUNNING_MODAL'}

class LoadsDecorator:
	is_installed = False
	handlers = []
	linear_load_shader = ShaderInfo("DistributedLoad")

	@classmethod
	def install(cls, context):
		if cls.is_installed:
			cls.uninstall()
		handler = cls()
		#cls.handlers.append(SpaceView3D.draw_handler_add(handler.draw_load_values, (context,), "WINDOW", "POST_PIXEL"))
		#cls.handlers.append(SpaceView3D.draw_handler_add(handler.draw_extra_info, (context,), "WINDOW", "POST_PIXEL"))
		#cls.handlers.append(
		#	SpaceView3D.draw_handler_add(handler.draw_curve_loads, (context,), "WINDOW", "POST_VIEW")
		#)
		cls.handlers.append(SpaceView3D.draw_handler_add(handler, (), "WINDOW", "POST_VIEW"))
		cls.is_installed = True

	@classmethod
	def uninstall(cls):
		for handler in cls.handlers:
			try:
				SpaceView3D.draw_handler_remove(handler, "WINDOW")
			except ValueError:
				pass
		cls.is_installed = False

	@classmethod
	def update(cls, context):
		cls.linear_load_shader.update()

	def __call__(cls):
		# set open gl configurations
		original_blend = gpu.state.blend_get()
		gpu.state.blend_set('ALPHA')
		original_depth_mask = gpu.state.depth_mask_get()
		gpu.state.depth_mask_set(True)
		original_depth_test = gpu.state.depth_test_get()
		gpu.state.depth_test_set('LESS')
		
		cls.draw_batch("DistributedLoad")

		# restore opengl configurations
		gpu.state.blend_set(original_blend)
		gpu.state.depth_mask_set(original_depth_mask)
		gpu.state.depth_test_set(original_depth_test)

	def draw_batch(cls,shader_type: str):
		""" param: shader_type: type of shader in ["DistributedLoad", "PointLoad"]"""
		
		shader = cls.get_shader("DistributedLoad")
		if shader_type == "DistributedLoad":
			shader_info = cls.linear_load_shader
			if not shader_info.is_empty:
				shader = shader_info.shader
				args = shader_info.args
				indices = shader_info.indices
				batch = batch_for_shader(shader, 'TRIS', args, indices=indices)
				matrix = bpy.context.region_data.perspective_matrix
				shader.uniform_float("viewProjectionMatrix", matrix)
				shader.uniform_float("spacing", 0.2) #todo make it customizable
				shader.uniform_float("arrow", 20) #todo make it customizable
				batch.draw(shader)

class ShaderInfo:
	def __init__(cls,shader_type: str):
		cls.is_empty = True
		cls.shader = None
		cls.shader_type = shader_type
		cls.args = {}
		cls.indices = []
	
	def update(cls):
		cls.get_shader()
		cls.get_args_and_indices()
		if len(cls.args["position"]):
			cls.is_empty = False
		
	def get_shader(cls):
	""" param: shader_type: type of shader in ["DistributedLoad", "PointLoad"]
		return: shader"""
		if cls.shader_type == "DistributedLoad":
			cls.shader_type = shader_type
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
		
			cls.shader = gpu.shader.create_from_info(shader_info)
			del vert_out
			del shader_info

	def get_args_and_indices(): #for now it only works for distributed loads
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
	
			#get shader args for each direction
			reference_frame = 'LOCAL_COORDS' #make it a scene property so it can be changed in a panel
			is_global = assert(reference_frame == 'GLOBAL_COORDS')
			x_match = assert(x_axis == Vector((1,0,0)))
			y_match = assert(y_axis == Vector((0,1,0)))
			z_match = assert(z_axis == Vector((0,0,1)))
			direction_dict = {
				"fx": Vector((1,0,0)) if is_global and not x_match else Vector((0,1,1)) if is_global else y_axis+z_axis,
				"fy": Vector((0,1,0)) if is_global and not y_match else Vector((1,0,1)) if is_global else y_axis,
				"fz": Vector((0,0,1)) if is_global and not z_match else Vector((1,1,0)) if is_global else z_axis,
				"mx": Vector((1,0,0)) if is_global and not x_match else Vector((0,1,1)) if is_global else y_axis+z_axis,
				"my": Vector((0,1,0)) if is_global and not y_match else Vector((1,0,1)) if is_global else y_axis,
				"mz": Vector((0,0,1)) if is_global and not z_match else Vector((1,1,0)) if is_global else z_axis,
			}
			xyzdict = get_loads_per_direction(activity_list,global_to_local)
			keys = ["fx","fy","fz","mx","my","mz"]
			maxforce = 200 #should be the maximum value expected for the loads
			for key in keys:
				polyline = xyzdict[key]["polyline"]
				sinus = xyzdict[key]["sinus"]
				quadratic = xyzdict[key]["quadratic"]
				constant = xyzdict[key]["constant"]
				direction = direction_dict[key] #depends on the key and on the frame of reference
				
				addindex = len(indices)
				counter = 0
				for i in range(len(polyline)-1):
					current = Vector(polyline[i]+[0])
					nextitem = Vector(polyline[i+1]+[0])
	
					if current[1] != 0 or nextitem[1] != 0: #if there is load in the z direction
						negative = -direction + start_co + x_axis*current.x
						positive = direction + start_co + x_axis*current.x
						coords.append(negative)
						coords_2d.append((current.x, maxforce))
						load_info.append((sinus, quadratic, current.y + constant))
						color.append((0,0,1))
	
						coords.append(positive)
						coords_2d.append((current[0],-maxforce))
						load_info.append((sinus, quadratic, current.y + constant))
						color.append((0,0,1))
	
						indices.append((0 + counter + addindex,
										1 + counter + addindex,
										2 + counter + addindex))
						indices.append((3 + counter + addindex,
										1 + counter + addindex,
										2 + counter + addindex))
						if i == len(polyline)-2:
							negative = -direction + start_co + x_axis*nextitem.x
							positive = direction + start_co + x_axis*nextitem.x
							coords.append(negative)
							coords_2d.append((nextitem.x, maxforce))
							load_info.append((sinus, quadratic, nextitem.y + constant))
							color.append((0,0,1))
	
							coords.append(positive)
							coords_2d.append((nextitem.x,-maxforce))
							load_info.append((sinus, quadratic, nextitem.y + constant))
							color.append((0,0,1))
	
						counter += 2
		cls.args = {"position": coords,"color": color, "sin_quad_lin_forces": load_info,"uv_coord": coords_2d}
		cls.indices = indices


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



def get_loads_per_direction(activity_list,global_to_local):
	""" returns a dict with values for applied loads in each direction
	 return = {
	 			"fx": values_in_this_direction
	 			"fy": values_in_this_direction
	 			"fz": values_in_this_direction
	 			"mx": values_in_this_direction
	  			"my": values_in_this_direction
	  			"mz": values_in_this_direction
		 		}
	 values_in_this_direction = {
  								"constant": float,
					  			"quadratic": float,
						 		"sinus": float,
								"polyline": list[(position: float, load: float),(position: float, load: float),...]
					   			}
	"""
	loads_dict = get_loads_dict(activity_list,global_to_local)
	const = loads_dict["constant forces"]
	quad = loads_dict["quadradic forces"]
	sinus = loads_dict["sinus forces"]
	loads = loads_dict["load configurations"]
	unique_list = getuniquepositionlist(loads)
	final_list = []
	for pos in unique_list:
		value = get_before_and_after(pos,loads)
		if value["before"] == value["after"]:
			final_list.append([pos]+value["before"])
		else:
			final_list.append([pos]+value["before"])
			final_list.append([pos]+value["after"])
	if len(final_list)>2:
		del final_list[0]
		del final_list[-1]
	array = np.array(final_list) #7xn -> ["pos","fx","fy","fz","mx","my","mz"]
	keys = ["fx","fy","fz","mx","my","mz"]
	polyline = {
	 			"fx": [],
	 			"fy": [],
	 			"fz": [],
	 			"mx": [],
	  			"my": [],
	  			"mz": [],
		 		}
	values_in_this_direction = {
  								"constant": 0,
					  			"quadratic": 0,
						 		"sinus": 0,
								"polyline": []
					   			}
	return_value = {
	 			"fx": values_in_this_direction,
	 			"fy": values_in_this_direction,
	 			"fz": values_in_this_direction,
	 			"mx": values_in_this_direction,
	  			"my": values_in_this_direction,
	  			"mz": values_in_this_direction,
		 		}
	for component, key in enumerate(keys):
		if( sinus[component] or quad[component] or const[component] or
			any(item != 0 for item in array[:,component+1])):
			for currentitem in final_list:
				polyline[key].append(currentitem[component+1])
		return_value[key]["constant"] = const[component+1]
		return_value[key]["quadratic"] = quad[component+1]
		return_value[key]["sinus"] = sinus[component+1]
		return_value[key]["polyline"] = polyline[key]
	
	return return_value


def getuniquepositionlist(load_config_list):
	"""return an ordereded list of unique locations based on the load configuration list
 ex: load_config_list = [[{"pos":1.0,...},{"pos":3.0,...}],
 						 [{"pos":2.0,...},{"pos":3.0,...}],
						 [{"pos":1.5,...},{"pos":2.5,...}]]
		return = [1.0, 1.5, 2.0, 2.5, 3.0]
 """
	unique = []
	for config in load_config_list:
		for info in config:
			if info["pos"] in unique:
				continue
			unique.append(info["pos"])
	unique.sort()
	return unique

def interp1d(l1,l2, pos):
	fac = (l2[1]-l1[1])/(l2[0]-l1[0])
	v = l1[1] + fac*(pos-l1[0])
	return v

def interpolate(pos,loadinfo,start,end,key):
	result = Vector((0,0,0))
	result
	for i in range(3):
		value1 = [loadinfo[start]["pos"], loadinfo[start][key][i]] #[position, force_component]
		value2= [loadinfo[end]["pos"], loadinfo[end][key][i]] #      [position, force_component]
		result[i] = interp1d(value1,value2, pos) #             interpolated [position, force_component]
	return result

def get_before_and_after(pos,load_config_list):
	""" get total values for forces and moments with polilyne distribution
 		before and after the position
 ex: load_config_list = [[{"pos":1.0,...,"forces":(1,0,0),...},{"pos":3.0,...,"forces":(3,0,0),...}],
 						 [{"pos":2.0,...,"forces":(1,0,0),...},{"pos":3.0,...,"forces":(1,0,0),...}],
						 [{"pos":1.5,...,"forces":(1,0,0),...},{"pos":2.5,...,"forces":(1,0,0),...}]]
	   	pos = 2.0
		return = {
  					"before": (3,0,0,0,0,0),  ->(fx, fy, fz, mx, my, mz)
	   				" after": (4,0,0,0,0,0)   ->(fx, fy, fz, mx, my, mz)
	  			}
 """
	force_before = Vector((0,0,0))
	force_after = Vector((0,0,0))
	moment_before = Vector((0,0,0))
	moment_after = Vector((0,0,0))

	for config in load_config_list:
		if pos < config[0]["pos"] or pos > config[-1]["pos"]:
			continue
		start = 0
		end = len(config)-1
		while end-start > 0:
			if pos < config[start]["pos"] or pos > config[end]["pos"]:
				break
			if config[start]["pos"] == pos:
				if config[start]["descr"] in ['start','middle']:
					force_after += config[start]["forces"]
					moment_after += config[start]["moments"]
				elif config[start]["descr"] in ['end','middle']:
					force_before += config[start]["forces"]
					moment_before += config[start]["moments"]

			elif config[end]["pos"] == pos:
				if config[end]["descr"] in ['start','middle']:
					force_after += config[end]["forces"]
					moment_after += config[end]["moments"]
				elif config[end]["descr"] in ['end','middle']:
					force_before += config[end]["forces"]
					moment_before += config[end]["moments"]

			elif end-start == 1:
				force_before += interpolate(pos,config,start,end,"forces")
				force_after += interpolate(pos,config,start,end,"forces")
				moment_before += interpolate(pos,config,start,end,"moments")
				moment_after += interpolate(pos,config,start,end,"moments")
			start += 1
			end -=1
	return_value = {
		"before": [force_before.x,force_before.y,force_before.z,
				   moment_before.x,moment_before.y,moment_before.z],
	   	"after": [force_after.x, force_after.y, force_after.z,
				  moment_after.x, moment_after.y, moment_after.z]
				  }
	return return_value

def get_loads_dict(activity_list,global_to_local):
	"""
	get load list
 	activity_list: list of IfcStructuralCurveAction or IfcStructuralCurveReaction applied in the structural curve member
	global_to_local: transformation matrix from global coordinates to local coordinetes
 	return: dict{
  				"constant force": (fx,fy,fz,mx,my,mz),	-> sum of linear loads applied with constant distribution
	   			"quadratic force": (fx,fy,fz,mx,my,mz),	-> sum of linear loads applied with quadratic distribution
	   			"sinus force": (fx,fy,fz,mx,my,mz),		-> sum of linear loads applied with sinus distribution
	   			"load configuration": list				-> list of load configurations for linear and polyline distributions of linear loads
	   			}
	   description of "load configuration":
	list[							-> one item (list)for each IfcStructuralCurveAction applied in the member with IfcStructuralLoadConfiguration as the applied load
 		list[						-> one item (dict) for each item found in the Locations attribute of IfcLoadConfiguration
   			dict{
	  			"loc": float,		-> local position along curve length
	  			"descr": string,	-> describe if the item is at the star, middle or end of the list
	  			"forces": Vector,	-> linear force applied at that point
	  			"moments": Vector	-> linear moment applied at that point
	  			}
	  		]
	 	]
		  """
	constant_force = Vector((0,0,0))
	constant_moment = Vector((0,0,0))
	quadratic_force = Vector((0,0,0))
	quadratic_moment = Vector((0,0,0))
	sinus_force = Vector((0,0,0))
	sinus_moment = Vector((0,0,0))
	load_configurations = []

	momentvalues = Vector((0,0,0))
	constvalues = [forcevalues,momentvalues]
	parabolavalues = [forcevalues,momentvalues]        
	sinusvalues = [forcevalues,momentvalues]
	load_values = []
	unit_scale = ifcopenshell.util.unit.calculate_unit_scale(tool.Ifc.get(),"LENGTHUNIT")

	def get_force_vector(load,transform_matrix):
		x = 0 if getattr(l, 'LinearForceX', 0) is None else getattr(l, 'LinearForceX', 0)
		y = 0 if getattr(l, 'LinearForceY', 0) is None else getattr(l, 'LinearForceY', 0)
		z = 0 if getattr(l, 'LinearForceZ', 0) is None else getattr(l, 'LinearForceZ', 0)
		return transform_matrix @ Vector((x,y,z))
	
	def get_moment_vector(load,transform_matrix):
		x = 0 if getattr(l, 'LinearMomentX', 0) is None else getattr(l, 'LinearMomentX', 0)
		y = 0 if getattr(l, 'LinearMomentY', 0) is None else getattr(l, 'LinearMomentY', 0)
		z = 0 if getattr(l, 'LinearMomentZ', 0) is None else getattr(l, 'LinearMomentZ', 0)
		return transform_matrix @ Vector((x,y,z))
	
	for activity in activity_list:
		load = activity.AppliedLoad
		global_or_local = activity.GlobalOrLocal
		reference_frame = 'LOCAL_COORDS' #make it a scene property so it can be changed in a panel
		transform_matrix = Matrix.identity()
		if reference_frame == 'LOCAL_COORDS' and global_or_local != reference_frame:
			transform_matrix = global_to_local
		elif reference_frame == 'GLOBAL_COORDS' and global_or_local != reference_frame:
			transform_matrix = global_to_local.inverted()
		#values for linear loads
		if load.is_a('IfcStructuralLoadConfiguration'):
			locations = getattr(load, 'Locations', [])
			values = [l for l in getattr(load, 'Values', None) 
					if l.is_a() == "IfcStructuralLoadLinearForce"
					]
			config_list = []
			for i,l in enumerate(values):
				forcevalues = get_force_vector(l)
				momentvalues = get_moment_vector(l)
				if i == 0:
					descr = 'start'
				elif i == len(values)-1:
					descr = 'end'
				else:
					descr = 'middle'
				config_list.append(
					{"loc": locations[i][0]*unit_scale,
					 "descr": descr,
					 "forces":forcevalues,
					 "moments":momentvalues}
				)
			load_configurations.append(config_list)
		else:
			forcevalues = get_force_vector(load)
			momentvalues = get_moment_vector(load)
			if 'CONST' == getattr(activity, 'PredefinedType', None):
				constant_force += forcevalues
				constant_moment += momentvalues
			elif 'PARABOLA' == getattr(activity, 'PredefinedType', None):
				quadratic_force += forcevalues
				quadratic_moment += momentvalues
			elif 'SINUS' == getattr(activity, 'PredefinedType', None):
				sinus_force += forcevalues
				sinus_moment += momentvalues
	return_value = {
  				"constant force": [constant_force.x,constant_force.y,constant_force.z,
						 		   constant_moment.x,constant_moment.y,constant_moment.z],
	   			"quadratic force": [quadratic_force.x,quadratic_force.y,quadratic_force.z,
						   			quadratic_moment.x,quadratic_moment.y,quadratic_moment.z],
	   			"sinus force": [sinus_force.x,sinus_force.y,sinus_force.z,
					   			sinus_moment.x,sinus_moment.y,sinus_moment.z],
	   			"load configuration": load_configurations
	   			}
	return return_value
