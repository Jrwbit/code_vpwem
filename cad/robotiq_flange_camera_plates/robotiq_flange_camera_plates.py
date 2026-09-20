"""UR5 flange sandwich camera plates for original Robotiq couplings.

Two parts are generated:
  * Hand-E plate: UR flange -> plate -> GRP-CPL-062 -> Hand-E.
  * FT 300-S plate: UR flange -> plate -> FTS-300-S-CPL-014 -> FT 300-S.

The robot-side pattern is ISO 9409-1-50-4-M6.  Camera plate supports D435
(M3, 45 mm centres) and D455 (M4, 95 mm centres).  For a working robot,
machine this part in 6061 aluminium; the STL is a fit-check prototype only.
"""

import math
import os

import FreeCAD as App
import Part


OUT = "/data/wjr/vpwem/code_vpwem/cad/robotiq_flange_camera_plates"
DOC_NAME = "Robotiq_Flange_Camera_Plates"

# UR5 / UR5e ISO 9409-1-50-4-M6 interface.
PLATE_THICKNESS = 3.0
CENTRE_CLEARANCE = 34.0
BOLT_PCD = 50.0
M6_CLEARANCE = 6.8
BOLT_PATTERN_DEG = 45.0
PIN_CLEARANCE = 6.6

# D400 rear mounting patterns.
D435_SPACING = 45.0
D435_CLEARANCE = 3.4
D455_SPACING = 95.0
D455_CLEARANCE = 4.5
CAMERA_PLATE_WIDTH = 140.0
CAMERA_PLATE_HEIGHT = 44.0


def z_hole(x, y, diameter, z0, depth):
    return Part.makeCylinder(diameter / 2.0, depth, App.Vector(x, y, z0))


def make_plate(disk_od, camera_x):
    outer_r = disk_od / 2.0
    shape = Part.makeCylinder(outer_r, PLATE_THICKNESS)
    shape = shape.cut(Part.makeCylinder(
        CENTRE_CLEARANCE / 2.0, PLATE_THICKNESS + 2, App.Vector(0, 0, -1)))

    bolt_r = BOLT_PCD / 2.0
    for index in range(4):
        angle = math.radians(BOLT_PATTERN_DEG + 90.0 * index)
        x, y = bolt_r * math.cos(angle), bolt_r * math.sin(angle)
        shape = shape.cut(z_hole(x, y, M6_CLEARANCE, -1, PLATE_THICKNESS + 2))
    # Circular locating-pin hole preserves repeatable coupling orientation.
    shape = shape.cut(z_hole(bolt_r, 0, PIN_CLEARANCE, -1,
                             PLATE_THICKNESS + 2))

    # Flat radial arm.  Camera back is on +Z; camera looks along tool +Z.
    # 12 mm radial overlap gives a true area union with the circular plate.
    arm_x0 = outer_r - 12.0
    arm = Part.makeBox(camera_x + CAMERA_PLATE_WIDTH / 2.0 - arm_x0,
                        CAMERA_PLATE_HEIGHT, PLATE_THICKNESS,
                        App.Vector(arm_x0, -CAMERA_PLATE_HEIGHT / 2.0, 0))
    shape = shape.fuse(arm, 0.01).removeSplitter()

    # D435 inner pair and D455 outer pair.  Heads recess on the robot side.
    for dx in (-D435_SPACING / 2.0, D435_SPACING / 2.0):
        shape = shape.cut(z_hole(camera_x + dx, 0, D435_CLEARANCE,
                                 -1, PLATE_THICKNESS + 2))
        shape = shape.cut(z_hole(camera_x + dx, 0, 7.0, -0.01, 2.0))
    for dx in (-D455_SPACING / 2.0, D455_SPACING / 2.0):
        shape = shape.cut(z_hole(camera_x + dx, 0, D455_CLEARANCE,
                                 -1, PLATE_THICKNESS + 2))
        shape = shape.cut(z_hole(camera_x + dx, 0, 8.5, -0.01, 2.0))
    return shape.removeSplitter()


def add_shape(doc, name, label, shape, color, view_y):
    obj = doc.addObject("PartDesign::Feature", name)
    obj.Label = label
    obj.Shape = shape.translated(App.Vector(0, view_y, 0))
    if obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = color
    return obj


def export_shape(doc, shape, path):
    obj = doc.addObject("PartDesign::Feature", "_ExportTemp")
    obj.Shape = shape
    Part.export([obj], path)
    doc.removeObject(obj.Name)


try:
    doc = App.getDocument(DOC_NAME)
except Exception:
    doc = None
if doc is None:
    doc = App.newDocument(DOC_NAME)
for obj in list(doc.Objects):
    if obj.Name.startswith("FlangePlate_"):
        doc.removeObject(obj.Name)

os.makedirs(OUT, exist_ok=True)
configs = [
    ("HandE", "Hand-E: UR flange / plate / original coupling", 86.0, 112.0,
     0.0, (0.12, 0.55, 0.90)),
    ("FT300S", "FT 300-S: UR flange / plate / original coupling", 100.0, 124.0,
     -160.0, (0.95, 0.45, 0.12)),
]
for short, label, disk_od, camera_x, view_y, color in configs:
    solid = make_plate(disk_od, camera_x)
    obj = add_shape(doc, "FlangePlate_" + short, label, solid, color, view_y)
    obj.addProperty("App::PropertyString", "Interface", "Design")
    obj.Interface = "ISO 9409-1-50-4-M6; original locating pin retained"
    obj.addProperty("App::PropertyString", "CameraHoles", "Design")
    obj.CameraHoles = "D435: 2x dia3.4 / 45 mm; D455: 2x dia4.5 / 95 mm"
    stem = "robotiq_%s_flange_camera_plate" % short.lower()
    export_shape(doc, solid, os.path.join(OUT, stem + ".step"))
    export_shape(doc, solid, os.path.join(OUT, stem + ".stl"))
    print(obj.Name, "valid=", solid.isValid(), "solids=", len(solid.Solids),
          "volume_mm3=", round(solid.Volume, 1))

doc.recompute()
doc.saveAs(os.path.join(OUT, "robotiq_d435_d455_flange_camera_plates.FCStd"))
print("Saved to", OUT)
