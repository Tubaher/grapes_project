import numpy as np
import glob
import os
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import normalize
from count_heatMap import *
from start_end_pts import *
"""
	The following dictionary has data associated with the quarter`s polygons. Each key represents one land.
	Inside each camp, there is a dictionary where each key represents a quarter. Inside each quarter there
	is a dictionary where the key "poly_verts" corresponds to a list with two elements; the first one is a list with the vertices coordinates of the side where the rows begin and the last one is a list with the
	vertices coordinates where the rows end. "diagonal" has as value a list with two points: the beginning
	point and the endpoint of the straight which will be used to build the subdivisions of the polygon. The
	last key is "hilera_angulo" which corresponds to the angle of the orientation of the rows with respect
	to the horizontal.
"""
COORDENADAS_POLY = {"2": {"2": {"poly_verts": [[(641, 38), (402, 38), (388, 613)], [(641, 38), (649, 610), (388, 613)]],
"diagonal": [(388, 613), (641, 38)], "hilera_angulo": 0, "n_hileras": 112}, "3": {"poly_verts": [[(908, 35), (662, 35), (672, 611)], [(908, 35), (932, 604), (672, 611)]],
"diagonal": [(672, 611), (908, 35)], "hilera_angulo": 0, "n_hileras": 112}, "4": {"poly_verts": [[(1134, 33), (928, 33), (960, 609)], [(1134, 33), (1282, 594), (1266, 612), (960, 609)]],
"diagonal": [(960, 609), (1134, 33)], "hilera_angulo": 0, "n_hileras": 113}},
					"3": {"1": {"poly_verts": [[(1192, 43), (450, 10), (437, 32), (374, 321), (360, 478), (402, 690)], [(1192, 43), (1183, 85), (1113, 161), (999, 125), (933, 211), (859, 352), (821, 612), (785, 638), (737, 652), (402, 690)]],
"diagonal": [(1192, 43), (402, 690)], "hilera_angulo": 20, "n_hileras": 259}}}


# Size, in pixels, of each segment in the row.
DELTA_H = 20
LINE_WIDTH = 1.8
TOP_BOUNDARY_COUNT = 200


def read_location_pickle(frames_cuartel, pickle_filename):
	"""
		Read a pickle file that contains the number of frames of each detection of a particular grape bunch of a quarter (it can be am or pm) and then, it inserts in the dictionary frames_cuartel with the row id and the corresponding hour.
	:param (dict) frames_cuartel:
       -Dictionary with the lists that contain the numbers of the frames of the detection which are ordered per row and hour.
	:param (string) pickle_filename:
		path to the pickle file.
	"""
	# Get the row id
	hilera_idx = int(pickle_filename.split("_")[-2])
	# if there is not an entry for the row in the dictionary frames_cuartel, a new one is created.
	if hilera_idx not in frames_cuartel:
		frames_cuartel[hilera_idx] = {}

	# The hour is extracted from the pickle name (am or pm).
	hora = int(pickle_filename.split("_")[-1].split(".")[0])

	# Load the pickle file
	with open(pickle_filename, "rb") as loc_pickle:
		# create a dictionary of the row of the pickle.
		# Loads the dictionary racimos_locations which are saved in splash uvas.
		hilera_dict = pickle.load(loc_pickle)

	# np array of dimsions len(hilera_dict) x 2
	frames = np.zeros((len(hilera_dict), 2))

	# Read each item in the pickle and then save the number of the frame in np array frames.
	for idx, (k, v) in enumerate(hilera_dict.items()):
		# {id_racimo:(frame_count, x_center)}
		# frame_count
		frames[idx][1] = v[0]
		# id_racimo
		frames[idx][0] = k

	# The dictionary is updated with the list of frames for the hour and the
	# number of the corresponding row

	frames_cuartel[hilera_idx][hora] = frames


def map_cuartel(pickles_dir, ax, DATOS):
	"""

		Draw the bunch densities based on the lists with the number of frames
		of saved detections generated from the pickle files in pickles_dir,
		for each row of a given quarter
	:param (string) pickles_dir:
		directory with the location pickles of the quarters
	:param (Axes) ax:
		ax, of matplotlib, where the heat map will be drawn
	"""
	# Extract field and barracks number on which the heat map will be drawn

	campo = pickles_dir.split("/")[-1].split("_")[1]
	cuartel = pickles_dir.split("/")[-1].split("_")[2]

	# Saves the number of the frames of the detections as lists for each hour of each row.
    # Return
    # dict with info frames_cuartel['hilera_id']['hora']
    #
    # Each frames_cuartel['hilera_id']['hora'] has
    #   frames variable which is an numpy array with two columns [id_racimo,framecount]
	frames_cuartel = {}
	df_cuartel = DATOS[(DATOS["CAMPO_ID"] == campo) &
	                    (DATOS["CUARTEL_ID"] == cuartel)]
	pkl_filenames = glob.glob(pickles_dir+"/*.pkl")
	for loc_pkl in pkl_filenames:
		read_location_pickle(frames_cuartel, loc_pkl)

	# Information of the polygon of the quarter
    # TODO: read from json
	poly_info = COORDENADAS_POLY[campo][cuartel]
	polygon_verts = poly_info["poly_verts"]  # list of list of points [[(x1,x2)]]
	polygon_diagonal = poly_info["diagonal"]
	angle = poly_info["hilera_angulo"]
	n_hileras = poly_info["n_hileras"]

	# Segment the diagonal line of the polygon
    # Return: list of points [(x1,y1)] of the segmented line
    # [star_point] + points + [end_point]
	pts_diagonal = split(*polygon_diagonal, n_hileras+1)

	# Find the starting and ending points of each row in the polygon
	start_points, end_points = find_polygon_divisions(
	    pts_diagonal, polygon_verts, angle)

	# The max and min id of the rows are extracted.
	hilera_max = max(list(frames_cuartel.keys()))
	hilera_min = min(list(frames_cuartel.keys()))
	# The start and end points are assigned to each individual row
	hilera_pts = {}
	for hilera_idx in range(hilera_min, hilera_max+1):
		hilera_pts[hilera_idx] = (
		    start_points[hilera_idx-hilera_min], end_points[hilera_idx-hilera_min])

	# Draw the density of bunches in each row
    # dict key: hilera_id value: value: dict{hora: { <frames array [id_racimo,framecount]>}}
	for key, value in frames_cuartel.items():
		# The density in the row is only drawn if both am and pm counts exist
		am_exist = 1 in value.keys()
		pm_exist = 2 in value.keys()
		if am_exist and pm_exist:
			am_frames = value[1][:, 1]
			am_ids = value[1][:, 0]
			pm_frames = value[2][:, 1]
			pm_ids = value[2][:, 0]
			# print(len(df_hilera_am)==len(am_ids))
			hilera_coord = hilera_pts[key]
			"""
			TODO1
			# This part performs the extraction of the frame limits, for binning,
			# based on the maximum and minimum of the row's frame lists.
			This is not necessarily true, since at the beginning or end of the video there may be
			no camera movement and therefore impairs the precision of the location of the cluster
			densities in the row. Hardcodear los números de frames donde empieza y termina el
			movimiento de la cámara en cada video y trabajar con esos es más preciso.
			"""
			am_frame_lim = (min(am_frames), max(am_frames))
			pm_frame_lim = (min(pm_frames), max(pm_frames))
			# Calculation of bunches by segment of the row, for am and pm
			counts_am, segments = count_intervals(
			    am_frames, hilera_coord, am_frame_lim, DELTA_H)
			counts_pm, _ = count_intervals(
			    pm_frames, hilera_coord, pm_frame_lim, DELTA_H)

			# In case the program was executed with the argument --area, the count will correspond
			# to the sum in each segment of the areas detected for the clusters present in that segment
			if args.area is not None:
				am_areas = df_cuartel[(df_cuartel["HILERA_ID"] == str(key)) & (
				    df_cuartel["AM/PM"] == "1")].RACIMO_AREA.to_numpy()
				pm_areas = df_cuartel[(df_cuartel["HILERA_ID"] == str(key)) & (
				    df_cuartel["AM/PM"] == "2")].RACIMO_AREA.to_numpy()

				count_area_am = np.zeros(len(counts_am))
				count_skp_am = 0
				for count_indx in range(len(counts_am)):  # [4,7,6]
					count = counts_am[count_indx]  # am_areas = [7]
					for area_indx in range(count_skp_am, count_skp_am+count):
						count_area_am[count_indx] += am_areas[area_indx]
					count_skp_am += count

				# print(areas_count_am)
				count_area_pm = np.zeros(len(counts_pm))
				count_skp_pm = 0
				for count_indx in range(len(counts_pm)):
					count = counts_pm[count_indx]
					for area_indx in range(count_skp_pm, count_skp_pm+count):
						count_area_pm[count_indx] += pm_areas[area_indx]
					count_skp_pm += count
			"""
			TODO2
			Determining if the sum of the am and pm counts is indeed the best operation to reflect the number of bunches
            actually present in the row.
			TODO3
			Both am and pm counting videos are assumed to start from the same place in the row.
			This is the beginning or the end, although it is not necessarily true. It remains to include
			the orientation to the name of each pickle to determine in what sense of the
			counts_am and counts_pm lists the counts have to be combined.
            am = [1st, 2nd, 3nd]
            pm = [3nd, 2nd, 1st]
			"""
			if args.area:
				counts_total = np.add(count_area_am, count_area_pm)
				counts_total = counts_total/5000  # Por que divide para 5000?
			else:
				counts_total = np.add(counts_am, counts_pm)
			# Draw the density of bunches in the row
			plot_heatmap(counts_total, segments, ax, LINE_WIDTH, newcmp, norm)
			# Add, every 4 rows, the corresponding identifier number
			if key % 4 == 0:
				ax.text(hilera_coord[1][0], hilera_coord[1][1], key, fontsize=6,
					color="white", verticalalignment='bottom')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(
		description='Genera Mapa de Calor de Racimos')
	parser.add_argument('--campo', required=True,
						metavar="path/to/loc_pickles_folder/",
						help="Path al directorio con las carpetas de cada cuartel")
	parser.add_argument('--img', required=True,
						metavar="path/to/satelite_image.jpg",
						help="Path a la imagen satelital del campo")
	parser.add_argument('--sat_info', required=False,
						metavar="path/to/metadata_sat_images.json",
						help="Path a la metadata de las imagenes satelitales")
	parser.add_argument('--area', required=False,
						help="Si este flag esta seteado, se cuenta la suma del area de cada segmento",
						action="store_true")
	parser.add_argument('--boundary', required=False,
						metavar="limite de conteo",
						help="Entero que define el límite de conteo de cada segmento de hilera")
	parser.add_argument('--delta_h', required=False,
						metavar="delta de segmento de hilera",
						help="Longitud, en pixeles, de cada segmento de una hilera")
	parser.add_argument('--linewidth', required=False,
						metavar="grosor de linea",
						help="Grosor de las hileras dibujadas")
	parser.add_argument('--megapk', required=True,
						metavar="path to the megapickle of the field",
						help="Megapickle of all locations pickles of a field")
	args = parser.parse_args()

    # Load the coordenadas poly from a json file
	if args.sat_info is not None:
		json_file = open(args.sat_info, 'rt')
		COORDENADAS_POLY = json.load(json_file.read())

	print(args.megapk)
	DATOS = pd.read_pickle(args.megapk)
	print(DATOS)

	# The following code plots the heat maps for each quarter in curicó
	fig,ax = plt.subplots()

	if args.boundary is not None:
		TOP_BOUNDARY_COUNT = int(args.boundary)
	cmap = cm.get_cmap('hot', 256)
	newcolors = cmap(np.linspace(0,1,TOP_BOUNDARY_COUNT))
	newcmp = ListedColormap(newcolors)
	boundary = list(range(0,TOP_BOUNDARY_COUNT))
	norm = BoundaryNorm(boundary, ncolors = len(boundary))

	if args.delta_h is not None:
		DELTA_H = int(args.delta_h)

	if args.linewidth is not None:
		LINE_WIDTH = float(args.linewidth)

	for cuartel in os.listdir(args.campo):
		map_cuartel(args.campo+"/"+cuartel,ax,DATOS)

	img_path =args.img
	img = plt.imread(img_path)
	divider = make_axes_locatable(ax)
	cbar_ticks = list(range(0,TOP_BOUNDARY_COUNT,int(TOP_BOUNDARY_COUNT/10)))
	cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp),\
		cax = divider.append_axes("right", size="5%", pad=0.05),\
		ticks=cbar_ticks)
	campo_name = args.campo.split("/")[-2]

	if args.area:
		cbar_labels = ["{:.2e}".format(i*5000) for i in cbar_ticks]
		ax.set_title("Mapa de Calor de Area de Racimos - Campo "+campo_name)
	else:
		cbar_labels = cbar_ticks
		ax.set_title("Mapa de Calor de Conteo de Racimos - Campo "+campo_name)
	cbar.set_ticklabels(cbar_labels)
	ax.imshow(img)

	output_location = 'stuff/heat_maps/'
	if not os.path.exists(output_location):
		os.makedirs(output_location)

	plt.savefig(output_location + campo_name + '.png')
	# plt.show()

