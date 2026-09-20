"""Non-load-path split-clamp camera mounts for Robotiq metal couplings.

Creates two assemblies:
  1. Hand-E coupling clamp: nominal coupling OD 75 mm.
  2. FT 300-S coupling clamp: nominal coupling OD 89 mm.

The camera plate accepts either D435 (two M3 holes, 45 mm centres) or
D455 (two M4 holes, 95 mm centres).  The printed clamp grips the exterior
of the original metal coupling; it must never be placed in the UR flange's
load-bearing interface.
"""

import math
import os

import FreeCAD as App
import Mesh
import Part


OUT = "/data/wjr/vpwem/code_vpwem/cad/robotiq_camera_clamps"
DOC_NAME = "Robotiq_D435_D455_SplitClamp_Mounts"

# Camera hardware.
D435_HOLE_SPACING = 45.0
D435_CLEARANCE = 3.4
D455_HOLE_SPACING = 95.0
D455_CLEARANCE = 4.5
PLATE_WIDTH = 140.0
PLATE_DEPTH = 44.0
PLATE_THICKNESS = 5.0

# Clamp hardware: use two M4x30 socket-head screws and two M4 nyloc nuts.
CLAMP_HEIGHT = 8.0
WALL = 4.0
DIAMETRAL_CLEARANCE = 0.8
EAR_WIDTH = 10.0
EAR_LENGTH = 12.0
M4_CLEARANCE = 4.5


def z_hole(x, y, diameter, z0, length):
    return Part.makeCylinder(diameter / 2.0, length, App.Vector(x, y, z0))


def make_half(coupling_od, positive_y, with_camera_plate, camera_x):
    """One half of a split clamp.  Positive half carries the camera plate."""
    inner_r = (coupling_od + DIAMETRAL_CLEARANCE) / 2.0
    outer_r = inner_r + WALL
    ring = Part.makeCylinder(outer_r, CLAMP_HEIGHT).cut(
        Part.makeCylinder(inner_r, CLAMP_HEIGHT + 2.0, App.Vector(0, 0, -1)))

    # Divide along the XZ plane, yielding two printable halves.
    if positive_y:
        clip = Part.makeBox(4 * outer_r, 2 * outer_r, CLAMP_HEIGHT + 2,
                            App.Vector(-2 * outer_r, 0, -1))
    else:
        clip = Part.makeBox(4 * outer_r, 2 * outer_r, CLAMP_HEIGHT + 2,
                            App.Vector(-2 * outer_r, -2 * outer_r, -1))
    half = ring.common(clip)

    # Two opposing ears provide a positive M4 clamp across the split line.
    ear_x = outer_r - 3.5
    for sign in (-1, 1):
        x = sign * ear_x - EAR_WIDTH / 2.0
        y = 0 if positive_y else -EAR_LENGTH
        ear = Part.makeBox(EAR_WIDTH, EAR_LENGTH, CLAMP_HEIGHT,
                           App.Vector(x, y, 0))
        half = half.fuse(ear)
        half = half.cut(Part.makeCylinder(
            M4_CLEARANCE / 2.0, 2 * EAR_LENGTH + 2,
            App.Vector(sign * ear_x, -EAR_LENGTH - 1, CLAMP_HEIGHT / 2.0),
            App.Vector(0, 1, 0)))

    if with_camera_plate:
        # Camera back plane is XY.  Camera looks in +Z, parallel to the tool Z axis.
        plate_z = CLAMP_HEIGHT + 4.0
        plate_x0 = camera_x - PLATE_WIDTH / 2.0
        plate = Part.makeBox(PLATE_WIDTH, PLATE_DEPTH, PLATE_THICKNESS,
                             App.Vector(plate_x0, -PLATE_DEPTH / 2.0, plate_z))

        # A broad web and two triangular ribs connect the plate to the metal-coupling clamp.
        web_x0 = outer_r - 6.0
        web = Part.makeBox(14.0, 22.0, plate_z + PLATE_THICKNESS,
                            App.Vector(web_x0, -11.0, 0))
        half = half.fuse(web).fuse(plate)
        for y0 in (-18.0, 6.0):
            wire = Part.makePolygon([
                # Deliberate 1 mm overlaps make this a single manifold FDM solid.
                App.Vector(web_x0, y0, CLAMP_HEIGHT - 1.0),
                App.Vector(web_x0, y0, plate_z + 1.0),
                App.Vector(web_x0 + 30.0, y0, plate_z + 1.0),
                App.Vector(web_x0, y0, CLAMP_HEIGHT - 1.0),
            ])
            half = half.fuse(Part.Face(wire).extrude(App.Vector(0, 12.0, 0)))

        # D435 inner pair: M3. D455 outer pair: M4.
        for dx in (-D435_HOLE_SPACING / 2.0, D435_HOLE_SPACING / 2.0):
            half = half.cut(z_hole(camera_x + dx, 0, D435_CLEARANCE,
                                   plate_z - 1, PLATE_THICKNESS + 2))
            half = half.cut(z_hole(camera_x + dx, 0, 7.0,
                                   plate_z - 0.01, 2.5))
        for dx in (-D455_HOLE_SPACING / 2.0, D455_HOLE_SPACING / 2.0):
            half = half.cut(z_hole(camera_x + dx, 0, D455_CLEARANCE,
                                   plate_z - 1, PLATE_THICKNESS + 2))
            half = half.cut(z_hole(camera_x + dx, 0, 8.5,
                                   plate_z - 0.01, 2.5))

    return half.removeSplitter()


def add_part(doc, name, label, shape, color, placement_y=0):
    obj = doc.addObject("PartDesign::Feature", name)
    obj.Label = label
    obj.Shape = shape.translated(App.Vector(0, placement_y, 0))
    if obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = color
    return obj


def make_reference_cylinder(doc, name, label, od, height, placement_y):
    obj = doc.addObject("PartDesign::Feature", name)
    obj.Label = label
    obj.Shape = Part.makeCylinder(od / 2.0, height).translated(
        App.Vector(0, placement_y, 0))
    if obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = (0.55, 0.55, 0.55)
        obj.ViewObject.Transparency = 80
    return obj


def export_shape(doc, shape, filename):
    """Export an unplaced shape so every part is centred at its own origin."""
    temp = doc.addObject("PartDesign::Feature", "_ExportTemp")
    temp.Shape = shape
    Part.export([temp], filename)
    doc.removeObject(temp.Name)


try:
    doc = App.getDocument(DOC_NAME)
except Exception:
    doc = None
if doc is None:
    doc = App.newDocument(DOC_NAME)

for obj in list(doc.Objects):
    if obj.Name.startswith("Clamp_") or obj.Name.startswith("Reference_"):
        doc.removeObject(obj.Name)

# Positions separate the assemblies in the FreeCAD viewport only.
configs = [
    ("HandE", "Hand-E coupling camera clamp (nominal OD 75 mm)", 75.0, 16.9,
     112.0, 0.0, (0.15, 0.55, 0.90)),
    ("FT300S", "FT 300-S coupling camera clamp (nominal OD 89 mm)", 89.0, 13.6,
     124.0, -160.0, (0.95, 0.45, 0.12)),
]

os.makedirs(OUT, exist_ok=True)
for short, label, od, ref_height, camera_x, y_offset, color in configs:
    top = make_half(od, True, True, camera_x)
    bottom = make_half(od, False, False, camera_x)
    top_obj = add_part(doc, "Clamp_%s_CameraHalf" % short,
                       label + " — camera half", top, color, y_offset)
    bottom_obj = add_part(doc, "Clamp_%s_PlainHalf" % short,
                          label + " — plain half", bottom, color, y_offset)
    make_reference_cylinder(doc, "Reference_%s_Coupling" % short,
                            "Reference only: original metal coupling", od,
                            ref_height, y_offset)

    # Export at the real origin, not at the viewport offset.
    stem = "robotiq_%s_split_clamp" % short.lower()
    export_shape(doc, top, os.path.join(OUT, stem + "_camera_half.step"))
    export_shape(doc, bottom, os.path.join(OUT, stem + "_plain_half.step"))
    export_shape(doc, top, os.path.join(OUT, stem + "_camera_half.stl"))
    export_shape(doc, bottom, os.path.join(OUT, stem + "_plain_half.stl"))

doc.recompute()
doc.saveAs(os.path.join(OUT, "robotiq_d435_d455_split_clamp_assembly.FCStd"))

for obj in doc.Objects:
    if obj.Name.startswith("Clamp_"):
        print(obj.Name, "valid=", obj.Shape.isValid(),
              "solids=", len(obj.Shape.Solids),
              "volume_mm3=", round(obj.Shape.Volume, 1))
print("Saved to", OUT)
