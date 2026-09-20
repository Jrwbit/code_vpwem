"""Parametric UR5/UR5e flange mount for RealSense D435 and D455.

Run inside FreeCAD (tested with FreeCAD 1.x):
    exec(open("/home/wjr/code/vpwem/cad/ur5_realsense_mount/ur5_realsense_mount.py").read())

All dimensions are millimetres.
"""

import os

import FreeCAD as App
import Mesh
import Part


# UR tool flange: ISO 9409-1-50-4-M6.
FLANGE_OUTER_DIAMETER = 76.0
FLANGE_THICKNESS = 8.0
FLANGE_CENTER_CLEARANCE = 32.5
FLANGE_BOLT_CIRCLE = 50.0
FLANGE_M6_CLEARANCE = 6.8
FLANGE_PATTERN_ROTATION_DEG = 45.0
POSITION_SLOT_WIDTH = 6.4
POSITION_SLOT_LENGTH = 9.0

# Universal rear camera plate.
CAMERA_PLATE_WIDTH = 136.0
CAMERA_PLATE_HEIGHT = 44.0
CAMERA_PLATE_THICKNESS = 5.0
CAMERA_PLATE_BOTTOM_Z = 7.0  # 1 mm overlap with flange base for a sound union.
CAMERA_HOLE_ROW_Z = 31.0

# RealSense official rear mounting patterns.
D435_M3_SPACING = 45.0
D435_M3_CLEARANCE = 3.4
D455_M4_SPACING = 95.0
D455_M4_CLEARANCE = 4.5

OUTPUT_DIR = "/home/wjr/code/vpwem/cad/ur5_realsense_mount"
DOC_NAME = "UR5_RealSense_Universal_Mount"


def through_y_hole(x, z, diameter):
    return Part.makeCylinder(
        diameter / 2.0,
        CAMERA_PLATE_THICKNESS + 2.0,
        App.Vector(x, -CAMERA_PLATE_THICKNESS / 2.0 - 1.0, z),
        App.Vector(0, 1, 0),
    )


def radial_capsule_slot(center_radius, width, length):
    """Create a radial through-slot along +X for the UR positioning pin."""
    r = width / 2.0
    x0 = center_radius - length / 2.0
    x1 = center_radius + length / 2.0
    slot = Part.makeBox(length, width, FLANGE_THICKNESS + 2.0,
                        App.Vector(x0, -r, -1.0))
    slot = slot.fuse(Part.makeCylinder(
        r, FLANGE_THICKNESS + 2.0, App.Vector(x0, 0, -1.0)))
    slot = slot.fuse(Part.makeCylinder(
        r, FLANGE_THICKNESS + 2.0, App.Vector(x1, 0, -1.0)))
    return slot


def build_mount():
    # Circular robot-side flange adapter.
    flange = Part.makeCylinder(FLANGE_OUTER_DIAMETER / 2.0, FLANGE_THICKNESS)
    flange = flange.cut(Part.makeCylinder(
        FLANGE_CENTER_CLEARANCE / 2.0, FLANGE_THICKNESS + 2.0,
        App.Vector(0, 0, -1.0)))

    bolt_radius = FLANGE_BOLT_CIRCLE / 2.0
    for index in range(4):
        angle = App.Units.Quantity(
            f"{FLANGE_PATTERN_ROTATION_DEG + index * 90.0} deg").getValueAs("rad")
        x = bolt_radius * __import__("math").cos(angle)
        y = bolt_radius * __import__("math").sin(angle)
        flange = flange.cut(Part.makeCylinder(
            FLANGE_M6_CLEARANCE / 2.0, FLANGE_THICKNESS + 2.0,
            App.Vector(x, y, -1.0)))

    flange = flange.cut(radial_capsule_slot(
        bolt_radius, POSITION_SLOT_WIDTH, POSITION_SLOT_LENGTH))

    # Plate normal is Y. Camera rear face sits against the -Y face, looking -Y.
    plate = Part.makeBox(
        CAMERA_PLATE_WIDTH,
        CAMERA_PLATE_THICKNESS,
        CAMERA_PLATE_HEIGHT,
        App.Vector(-CAMERA_PLATE_WIDTH / 2.0,
                   -CAMERA_PLATE_THICKNESS / 2.0,
                   CAMERA_PLATE_BOTTOM_Z),
    )

    for x in (-D435_M3_SPACING / 2.0, D435_M3_SPACING / 2.0):
        plate = plate.cut(through_y_hole(x, CAMERA_HOLE_ROW_Z,
                                        D435_M3_CLEARANCE))
    for x in (-D455_M4_SPACING / 2.0, D455_M4_SPACING / 2.0):
        plate = plate.cut(through_y_hole(x, CAMERA_HOLE_ROW_Z,
                                        D455_M4_CLEARANCE))

    # Two broad triangular ribs on the robot side of the camera plate.
    ribs = None
    for center_x in (-34.0, 34.0):
        profile = Part.makePolygon([
            App.Vector(center_x - 4.0, CAMERA_PLATE_THICKNESS / 2.0, 7.0),
            App.Vector(center_x - 4.0, 29.0, 7.0),
            App.Vector(center_x - 4.0, CAMERA_PLATE_THICKNESS / 2.0, 36.0),
            App.Vector(center_x - 4.0, CAMERA_PLATE_THICKNESS / 2.0, 7.0),
        ])
        rib = Part.Face(profile).extrude(App.Vector(8.0, 0, 0))
        ribs = rib if ribs is None else ribs.fuse(rib)

    return flange.fuse(plate).fuse(ribs).removeSplitter()


doc = App.ActiveDocument
if doc is None or doc.Name != DOC_NAME:
    existing = App.getDocument(DOC_NAME) if DOC_NAME in App.listDocuments() else None
    doc = existing or App.newDocument(DOC_NAME)

old = doc.getObject("UniversalMount")
if old is not None:
    doc.removeObject(old.Name)

mount = doc.addObject("PartDesign::Feature", "UniversalMount")
mount.Label = "UR5 + RealSense D435/D455 universal mount"
mount.Shape = build_mount()
mount.addProperty("App::PropertyString", "RobotPattern", "Mounting")
mount.RobotPattern = "ISO 9409-1-50-4-M6; 4x dia 6.8 on PCD 50"
mount.addProperty("App::PropertyString", "D435Pattern", "Mounting")
mount.D435Pattern = "2x dia 3.4; 45 mm centres; use M3x8 max 0.4 Nm"
mount.addProperty("App::PropertyString", "D455Pattern", "Mounting")
mount.D455Pattern = "2x dia 4.5; 95 mm centres; use M4x8 max 0.4 Nm"
if mount.ViewObject is not None:
    mount.ViewObject.ShapeColor = (0.18, 0.52, 0.82)

doc.recompute()
os.makedirs(OUTPUT_DIR, exist_ok=True)
doc.recompute()
doc.saveAs(os.path.join(OUTPUT_DIR, "ur5_realsense_d435_d455_mount.FCStd"))
Part.export([mount], os.path.join(OUTPUT_DIR, "ur5_realsense_d435_d455_mount.step"))
stl_path = os.path.join(OUTPUT_DIR, "ur5_realsense_d435_d455_mount.stl")
Part.export([mount], stl_path)

mesh = Mesh.Mesh(stl_path)

print("Created:", mount.Label)
print("Shape valid:", mount.Shape.isValid(),
      "solids:", len(mount.Shape.Solids),
      "volume_mm3:", round(mount.Shape.Volume, 1))
print("STL solid:", mesh.isSolid(), "facets:", mesh.CountFacets)
print("Envelope mm:",
      round(mount.Shape.BoundBox.XLength, 2),
      round(mount.Shape.BoundBox.YLength, 2),
      round(mount.Shape.BoundBox.ZLength, 2))
print("Files exported to:", OUTPUT_DIR)
