import gmsh

gmsh.initialize()
gmsh.model.add("3d_lshape_quad_surface")

# 1. Definisci i due volumi
box1 = gmsh.model.occ.addBox(0, 0, 0, 2, 2, 6)
box2 = gmsh.model.occ.addBox(0, 0, 0, 2, 4, 2)

# 2. FUSIONA (non frammenta!) i volumi, rimuovendo gli oggetti originali
#    in modo da eliminare ogni interfaccia interna duplicata :contentReference[oaicite:0]{index=0}
fused, _ = gmsh.model.occ.fuse(
    [(3, box1)], [(3, box2)],
    removeObject=True, removeTool=True
)

# 3. Rimuovi eventuali entità duplicate residue :contentReference[oaicite:1]{index=1}
gmsh.model.occ.removeAllDuplicates()

# 4. Sincronizza il modello CAD in Gmsh
gmsh.model.occ.synchronize()

# 5. Recupera tutte le superfici (dim=2) e imposta il recombine
#    superficie per superficie (utile in geometrie concave) :contentReference[oaicite:2]{index=2}
surfaces = gmsh.model.getEntities(dim=2)
for _, tag in surfaces:
    gmsh.model.mesh.setRecombine(2, tag)

# 6. Imposta opzioni di meshing
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)   # formato ASCII 2.2
gmsh.option.setNumber("Mesh.Binary",       0)       # ASCII
gmsh.option.setNumber("Mesh.RecombineAll", 1)       # ricombina tutte le facce
gmsh.option.setNumber("Mesh.Algorithm",    8)       # algoritmo frontal-Delaunay
gmsh.option.setNumber("Mesh.MeshSizeMax",  0.0625)     # dimensione massima

# 7. (Opcionale) assegna physical groups dopo la sincronizzazione
for i, (_, tag) in enumerate(surfaces, start=1):
    gmsh.model.addPhysicalGroup(2, [tag], i)

# 8. Genera solo la mesh 2D superficiale di quads :contentReference[oaicite:3]{index=3}
gmsh.model.mesh.generate(2)

gmsh.fltk.run()

# 9. Esporta e chiudi
gmsh.write("lshape_mesh_4.msh")
gmsh.finalize()
