import numpy as np
import math
import matplotlib.pyplot as plt

def split(start_pt, end_pt, segments):
    """
        Segmenta la linea que va desde el punto start_pt a end_pt en la
        cantidad de segmentos definida por segments
    :param (tuple) start_pt:
        punto de partida
    :param (tuple) end_pt:
        punto final
    :param (int) segments:
        cantidad de segmentos que se desean
    :return (list(tuple)):
        lista con todos los puntos de las divisiones de los segmentos de la
        linea
    """
    # Define el incremento que hay que hacer en el eje x y eje y por cada segmento
    x_delta = (end_pt[0] - start_pt[0]) / float(segments)
    y_delta = (end_pt[1] - start_pt[1]) / float(segments)
    points = []
    # Añade cada punto de cada segmento a la lista points
    for i in range(1, segments):
        points.append((round(start_pt[0] + i * x_delta), round(start_pt[1] + i * y_delta)))
    # Agrega puntos inicial y final a points y retorna esto
    return [start_pt] + points + [end_pt]

def angle_line(point, angle, length):
    """
        Genera una recta en el punto point con un ángulo igual a
        angle y una longitud de 2*length. Retorna el punto inicial
        y el final de la recta
    :param (tuple) point:
        punto donde se quiere generar la recta
    :param (number) angle:
        ángulo de la recta (con respecto a la horizontal)
    :param (int) length:
        longitud de cada brazo de la recta
    :return (tuple,tuple):
        el punto inicial y el final de la recta
    """
    x, y = point
    # Define cual es el desface (absoluto) de los límite
    # de la recta con respecto al punto origen
    delta_y = length * math.sin(math.radians(angle))
    delta_x = length * math.cos(math.radians(angle))
    return (x-delta_x,y-delta_y),(x+delta_x,y+delta_y)

def line_intersection(line1, line2):
    """
        Encuentra la intersección entre las rectas generadas por los pares
        de puntos line1 y line2, si es que lo hay
    :param (tuple(tuple)) line1:
        par de puntos de primera recta
    :param (tuple(tuple)) line2:
        par de puntos de segunda recta
    :return (int,int) o (None):
        las coordenadas x e y de la intersección. Si no hay intersección
        retorna un None
    """
    # Define las diferencias de coordenadas entre los puntos de cada par
    # de puntos de las lineas
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        """
            Retorna la determinante entre la matriz formada
            por la concatenación de los vectores a y b
        :param (tuple) a:
            primer vector
        :param (tuple) b:
            segundo vector
        :return (int):
            la determinante
        """
        return a[0] * b[1] - a[1] * b[0]

    # Define la determinante entre los deltas de coordenadas de
    # cada par de puntos
    div = det(xdiff, ydiff)
    if div == 0:
       return None
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def check_in_segment(pt,line):
    """
        Chequea si es que el punto pt pertenece al segmento de linea que une
        el par de puntos line
    :param (tuple) pt:
        punto cuya pertenencia a segmento se quiere verificar
    :param (tuple(tuple)) line:
        par de puntos que define el segmento de linea
    :return (bool):
        si el punto pertenece o no al segmento de linea
    """
    # Este parámetro define un margen de error (pixeles) para verificar la pertenencia del
    # punto en el segmento
    error = 1
    # Verifica si las coordenadas en x del punto e encuentran en el rango horizontal definido
    # por el par de puntos
    check1 = min(line[0][0], line[1][0])-error  <= pt[0] <= max(line[0][0], line[1][0])+error
    # Verifica si las coordenadas en y del punto e encuentran en el rango vertical definido
    # por el par de puntos
    check2 = min(line[0][1], line[1][1])-error  <= pt[1] <= max(line[0][1], line[1][1])+error
    # Se cumple pertenencia si ambas verificaciones de rango se cumplen
    return check1 and check2


def find_polygon_divisions(line_pts, polygon_pts, angle):
    """
        Esta función primero genera una recta, con un ángulo definido por angle, en cada
        punto contenido dentro de line_pts. Luego, para cada recta generada, encuentra la intersección
        con cada mitad del polígono definido en polygon_pts. Retorna los puntos de intersección encontrados
    :param list(tuples) line_pts:
        puntos a partir de los cuales se generan las rectas de intersección
    :param list(list(tuples),list(tuples)) polygon_pts:
        vértices del polígono almacenados como una lista de dos listas. la primera de estas listas
        contiene los vértices del polígono de los lados donde empiezan las hileras y la segunda de
        estas listas los vértices del polígono de los lados donde terminan las hileras
    :param (number) angle:
        ángulo, con respecto a la horizontal, con la cual se generan las rectas de intersección
    :return list(tuples), list(tuples):
        las coordenadas de los puntos de intersección encontrados para cada lado del polígono
    """
    # Vértices de los lados donde empiezan las hileras
    start_poly_vertices = polygon_pts[0]
    # Vértices de los lados donde terminan las hileras
    end_poly_vertices = polygon_pts[1]
    # Largo de los brazos de cada recta de intersección generada
    line_length = 5000
    # Se inicializan los primeros par de vértices, tanto para los vértices finales
    # como iniciales, donde se evaluará intersección
    start_v_line = (start_poly_vertices[0],\
        start_poly_vertices[1])
    end_v_line = (end_poly_vertices[0],\
        end_poly_vertices[1])
    start = []
    end = []
    # Las variables siguientes se ocupan para ir cambiando los pares de puntos de cada lado
    # en caso de no encontrar intersección
    start_pt_indx = 0
    end_pt_indx = 0
    for pt in line_pts:
        # Genera una recta con ángulo angle y de longitud 2*length sobre el punto pt
        pt_line = angle_line((pt[0],pt[1]),angle,line_length)
        # Encuentra la intersección de la recta anterior con el lado donde empiezan las hileras
        # y el lado donde terminan
        start_intersection = (line_intersection(start_v_line,pt_line))
        end_intersection = (line_intersection(end_v_line,pt_line))
        start_sign = 1
        # Mientras no se encuentre alguna intersección con el lado de inicio o la intersección
        # encontrada no pertenezca al perímetro del polígono, se sigue intentando con otro lado
        # de inicio
        while start_intersection is None or not check_in_segment(start_intersection,start_v_line):
            # Indice del primer vertice del lado inicial
            start_pt_indx += 1*start_sign
            # Si el índice es igual o mayor al largo de la lista de vértices, se invierte
            # el sentido de exploración de esta. Lo mismo para si es menor a 0
            if start_pt_indx >= len(start_poly_vertices)-1:
                start_pt_indx -=2
                start_sign = -1
            elif start_pt_indx < 0:
                start_pt_indx += 2
                start_sign = 1
            # Actualiza el lado inicial para el cual se evaluará la intersección
            start_v_line = (start_poly_vertices[start_pt_indx],\
                            start_poly_vertices[start_pt_indx+1])
            start_intersection = (line_intersection(start_v_line,pt_line))
        end_sign = 1
        # Hace lo mismo que el loop anterior pero para los lados de fin de la hilera
        while end_intersection is None or not check_in_segment(end_intersection,end_v_line):
            end_pt_indx += 1*end_sign
            if end_pt_indx >= len(end_poly_vertices)-1:
                end_pt_indx -=2
                end_sign=-1
            elif end_pt_indx < 0:
                end_pt_indx += 2
                end_sign = 1
            end_v_line = (end_poly_vertices[end_pt_indx],\
                        end_poly_vertices[end_pt_indx+1])
            end_intersection = (line_intersection(end_v_line,pt_line))
        # Agrega el punto de intersección con el lado inicial y con el lado final
        # de la hilera a las lista correspondiente
        start.append(start_intersection)
        end.append(end_intersection)
    # Elimina los puntos de intersección encontrados para los puntos iniciales
    # y finales de cada parte del polígono (generan hileras muy chicas)
    start = start[1:-1]
    end = end[1:-1]
    return start, end

if __name__ == '__main__':
    # Algunas pruebas
    polygon = [[(641,38),(402,38),(388,613)],[(641,38),(649,610),(388,613)]]
    hilera_or = [polygon[0][2],polygon[1][0]]
    pts = split(hilera_or[0],hilera_or[1],113)
    start, end = find_polygon_divisions(pts,polygon,70)
    print(len(start))
    print(len(end))
    fig,ax = plt.subplots()
    for pt in start:
        ax.plot(start_pt[0],start_pt[1],"*")
    for pt in end:
        ax.plot(end_pt[0],end_pt[1],"*")
     
    img = plt.imread("./satImages/curicoMap.jpg")
    ax.imshow(img)
    plt.show()