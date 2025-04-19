// Cubed Sphere .geo file
// Crea un cubo strutturato con mesh transfinita e lo proietta radialmente sulla sfera unitaria

// Parametri
lc   = 1;
nDiv = 20;

// Punti (vertici del cubo)
Point(1) = {-1, -1, -1, lc};
Point(2) = { 1, -1, -1, lc};
Point(3) = { 1,  1, -1, lc};
Point(4) = {-1,  1, -1, lc};
Point(5) = {-1, -1,  1, lc};
Point(6) = { 1, -1,  1, lc};
Point(7) = { 1,  1,  1, lc};
Point(8) = {-1,  1,  1, lc};

// Linee (spigoli del cubo)
Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};

Line(5)  = {5, 6};
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};

Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Facce (Curve Loop e Plane Surface)
// Faccia inferiore: 1-2-3-4
Curve Loop(13) = {1,2,3,4};
Plane Surface(14) = {13};

// Faccia superiore: 5-6-7-8
Curve Loop(15) = {5,6,7,8};
Plane Surface(16) = {15};

// Faccia frontale: 1-2-6-5
// (da 1 a 2: Line1, da 2 a 6: Line10, da 6 a 5: inverso di Line5, da 5 a 1: inverso di Line9)
Curve Loop(17) = {1,10,-5,-9};
Plane Surface(18) = {17};

// Faccia posteriore: 4-3-7-8
// (da 4 a 3: inverso di Line3, da 3 a 7: Line11, da 7 a 8: Line7, da 8 a 4: inverso di Line12)
Curve Loop(19) = {-3,11,7,-12};
Plane Surface(20) = {19};

// Faccia sinistra: 1-5-8-4
// (da 1 a 5: Line9, da 5 a 8: inverso di Line8, da 8 a 4: inverso di Line12, da 4 a 1: Line4)
Curve Loop(21) = {9,-8,-12,4};
Plane Surface(22) = {21};

// Faccia destra: 2-3-7-6
// (da 2 a 3: Line2, da 3 a 7: Line11, da 7 a 6: inverso di Line6, da 6 a 2: inverso di Line10)
Curve Loop(23) = {2,11,-6,-10};
Plane Surface(24) = {23};

// Mesh transfinito: nDiv+1 nodi per ogni linea
Transfinite Line {1,2,3,4,5,6,7,8,9,10,11,12} = nDiv+1;

// Imposta la meshing transfinito (strutturato) per ogni faccia
Transfinite Surface {14} = {1,2,3,4};
Transfinite Surface {16} = {5,6,7,8};
Transfinite Surface {18} = {1,2,6,5};
Transfinite Surface {20} = {4,3,7,8};
Transfinite Surface {22} = {1,5,8,4};
Transfinite Surface {24} = {2,3,7,6};

// Ricombina per ottenere quadrilateri
Recombine Surface {14,16,18,20,22,24};

// Genera la mesh 2D
Mesh 2;

// Proiezione radiale: per ogni nodo (X,Y,Z) viene calcolato (X,Y,Z)/sqrt(X²+Y²+Z²)
// NOTA: Qui si usa il blocco Mesh.Move con il sottocomando Transform
Mesh.Move {
  Nodes { All; };
  Transform { "X/sqrt(X*X+Y*Y+Z*Z)", "Y/sqrt(X*X+Y*Y+Z*Z)", "Z/sqrt(X*X+Y*Y+Z*Z)" }
};

Save "cubed_sphere.msh";

