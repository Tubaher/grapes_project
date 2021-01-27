import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm


def segment_hilera(pt1, pt2, delta_h):
	"""
		Segmenta la hilera que va desde pt1 a pt2 basado en el tamaño
		de intervalo delta_h
	:param (int) pt1:
		punto de inicio de hilera
	:param (int) pt2:
		punto de fin de hilera
	:param (int) delta_h:
		tamaño de cada intervalo en pixeles
	:return (int,list(int),list(int)):
		retorna número de intervalos, las coordenadas horizontales y verticales
		de la hilera segmentada
	"""
	# Primero calcula la hipotenusa del triangulo que se forma entre las
	# coordenadas de inicio y fin de la hilera y la horizontal
	hipotenuse = np.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)
	# Encuentra el número de intervalos
	n_intervals = int(np.ceil(hipotenuse/delta_h))
	# Calcula el tamaño en x e y de cada intervalo
	x_delta = (pt2[0]-pt1[0])/float(n_intervals)
	y_delta = (pt2[1]-pt1[1])/float(n_intervals)
	# Crea las listas con las coordenadas x e y de cada punto
	# en la hilera segmentada
	x_hilera = []
	y_hilera = []
	x_hilera.append(pt1[0])
	y_hilera.append(pt1[1])
	for i in range(1,n_intervals):
		x_hilera.append(round(pt1[0] + i * x_delta))
		y_hilera.append(round(pt1[1] + i * y_delta))
	# El último intervalo puede no ser del mismo tamaño que el resto
	# por lo que se agrega al final
	x_hilera.append(pt2[0])
	y_hilera.append(pt2[1])
	return n_intervals, x_hilera, y_hilera

def create_frame_intervals(start_frame,end_frame,n_intervals):
	"""
		Segmenta el espacio de los números de frames de acuerdo a n_intervals
		para realizar el conteo de racimos por intervalo
	:param start_frame:
		frame donde inicia la hilera
	:param end_frame:
		frame donde termina la hilera
	:param n_intervals:
		cantidad de intervalos para segmentar
	:return list(int):
		Lista con espacio de números de frames segmentado
	"""
	# Calcula la cantidad de frames totales y extrae el tamaño, en frames, de los
	# intervalos (no aplica necesariamente para el último)
	total_frames = end_frame - start_frame
	delta_frame = int(np.floor(total_frames/n_intervals))
	# Segmenta el rango de frames desde start frame a end frame en base a
	# delta_frame
	frame_intervals = np.zeros(n_intervals+1)
	frame_intervals[0] = start_frame
	frame_intervals[n_intervals] = end_frame
	for i in range(1,n_intervals):
		frame_intervals[i] = start_frame + delta_frame*i
	return frame_intervals

def count_intervals(frames,pts_hilera,frame_limits,DELTA_H):
	"""
		Cuenta la cantidad de racimos que hay por cada segmento de la hilera
		de acuerdo a los números de los frames en el video donde se hicieron las
		detecciones
	:param list(int) frames:
		lista con los números de los frames de las detecciones
	:param tuple(tuple,tuple) pts_hilera:
		tupla con las coordenadas de los puntos de inicio y fin de la hilera
	:param tuple(int,int) frame_limits:
		tupla con los frames donde aparece el inicio y fin de la hilera
	:param (int) DELTA_H:
		tamaño, en pixeles, de los segmentos de la hilera
	:return:
	"""
	# Extracción de las coordenadas de inicio y fin de la hilera y
	# del número del primer y ultimo frame
	pt1, pt2 = pts_hilera[0], pts_hilera[1]
	start_frame, end_frame = frame_limits[0], frame_limits[1]
	# Segmenta la hilera de acuerdo a DELTA_H y retorna las coordenadas
	# de cada punto junto a la cantidad de intervalos
	n_intervals, x_hilera, y_hilera = segment_hilera(pt1,pt2,DELTA_H)
	# Segmenta el rango de frames de acuerdo al mismo número de intervalos
	# anterior
	frame_bins = create_frame_intervals(start_frame,end_frame,n_intervals)
	# Cuenta la cantidad de racimos por segmento de la hilera
	counts,_ = np.histogram(frames,bins=frame_bins)
	# Ordena los puntos de la hilera en intervalos para plotear
	points = np.array([x_hilera, y_hilera]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	return counts, segments

def plot_heatmap(counts,segments,ax,linewidth,colmap,norm):
	"""
		Dibuja la densidad de racimos en una hilera de un cuartel sobre ax
	:param (list) counts:
		conteos de cada racimo por segmento
	:param (list(tuple)) segments:
		intervalos de coordenadas de los segmentos de la hilera
	:param (Axes) ax:
		ax (matplotlib) donde se dibuja el mapa de calor
	:param (ColorMap) colmap:
		ColorMap (matplotlib) para los colores del mapa de calor
	:param (Norm) norm:
		Norm (matplotlib) que define como se comporta cada segmento
	:return:
	"""
	lc = LineCollection(segments, cmap=colmap, norm=norm)
	lc.set_array(counts)
	lc.set_linewidth(linewidth)
	line = ax.add_collection(lc)
	
def main():
	# Algunos tests hechos para probar el correcto funcionamiento
	# del código
	test_pt1 = (186,166)
	test_pt2 = (327,372)
	test_pt11 = (280,94)
	test_pt22 = (438,327)
	test_pts = [(test_pt1,test_pt2),(test_pt11,test_pt22)]
	test_start_frame_1 = 10
	test_end_frame_1 = 100
	test_start_frame_2 = 0
	test_end_frame_2 = 200
	test_frame_pts = [(test_start_frame_1,test_end_frame_1),\
	(test_start_frame_2,test_end_frame_2)]
	test_frames_1 = [5,40,20,30,70,100,56,20,50,90,100,20,45,2,70,24,87,39,44,\
	34,67,57,47,35,10,12,24,67,58,68,89,79,79,90]
	test_frames_2 = [20,30,20,30,120,200,56,20,51,110,130,200,125,75,76,95,101,139,144,\
	134,67,117,47,135,10,112,24,167,58,68,189,79,179,190]
	test_frames = [test_frames_1,test_frames_2]

	fig,ax = plt.subplots()
	for hilera_idx in range(len(test_pts)):
		counts,segments = count_intervals(test_frames[hilera_idx],\
			test_pts[hilera_idx],test_frame_pts[hilera_idx])
		plot_heatmap(counts,segments,ax)

	img = plt.imread("./vinaBack.png")
	ax.imshow(img)
	plt.show()

if __name__ == '__main__':
	main()
