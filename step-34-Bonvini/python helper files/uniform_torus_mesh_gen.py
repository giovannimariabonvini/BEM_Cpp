#!/usr/bin/env python3
import gmsh
import math

gmsh.initialize()
gmsh.model.add("torus_linear_quads_no_subdivision")

# Parametri del toro
R = 6.0  # Raggio maggiore
r = 2.0  # Raggio minore

mesh_size = 0.125

# Discretizzazione
Ntheta = int(12/mesh_size)  # suddivisioni lungo la circonferenza maggiore
Nphi   = int(8/mesh_size)   # suddivisioni lungo la circonferenza minore

# Lunghezza caratteristica (usata per i punti in Gmsh)
lc = 1.0

# Funzione per mappare la coppia (i, j) in un ID univoco per i punti Gmsh
def point_id(i, j):
    # useremo la formula i*(Nphi+1) + j + 1
    return i*(Nphi+1) + j + 1

#
# 1) Creazione dei punti
#
for i in range(Ntheta+1):
    theta = 2.0*math.pi*i/Ntheta
    for j in range(Nphi+1):
        phi = 2.0*math.pi*j/Nphi
        # coordinate 3D
        x = (R + r*math.cos(phi)) * math.cos(theta)
        y = (R + r*math.cos(phi)) * math.sin(theta)
        z = r*math.sin(phi)
        pid = point_id(i, j)
        gmsh.model.geo.addPoint(x, y, z, lc, pid)

#
# 2) Creazione delle superfici quadrangolari
#
# Ogni cella parametrica (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)
# con wrap in i e j.
quad_surfaces = []
for i in range(Ntheta):
    for j in range(Nphi):
        i1 = (i + 1) % Ntheta
        j1 = (j + 1) % Nphi

        p0 = point_id(i,   j)
        p1 = point_id(i1,  j)
        p2 = point_id(i1, j1)
        p3 = point_id(i,  j1)

        # Crea le 4 linee del quadrilatero
        l1 = gmsh.model.geo.addLine(p0, p1)
        l2 = gmsh.model.geo.addLine(p1, p2)
        l3 = gmsh.model.geo.addLine(p2, p3)
        l4 = gmsh.model.geo.addLine(p3, p0)

        # Forziamo ciascuna linea a non suddividersi (transfinite con 2 nodi)
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(l2, 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(l4, 2)

        # Creiamo un line loop e la superficie piana
        loop_id = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf_id = gmsh.model.geo.addPlaneSurface([loop_id])
        quad_surfaces.append(surf_id)

        # Facciamo in modo che la superficie sia "transfinite" con i 4 corner
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_id)

        # E chiediamo di ricombinare la mesh (anche se qui è già un quad perfetto)
        gmsh.model.geo.mesh.setRecombine(2, surf_id)

# Sincronizza la geometria col kernel di Gmsh
gmsh.model.geo.synchronize()

# Raggruppiamo tutte le superfici in un Physical Group (opzionale)
gmsh.model.addPhysicalGroup(2, quad_surfaces, 1)
gmsh.model.setPhysicalName(2, 1, "TorusSurfaceQuad")

#
# 3) Generazione mesh 2D
#
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombina in generale
gmsh.model.mesh.generate(2)

# Salviamo la mesh in .msh
gmsh.write("torus_mesh_4.msh")

gmsh.fltk.run()
gmsh.finalize()
