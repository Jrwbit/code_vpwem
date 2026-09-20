"""Printable UR5 M8 visual-detection target.

This is a *vision fixture*, not a load-bearing or thread-engagement tool.  It
bolts to the UR5/UR5e ISO 9409-1-50-4-M6 tool flange and presents an upright
M8-like bolt: 13 mm AF hex head plus an 8 mm shank with cosmetic M8x1.25
thread crests.  Dimensions are millimetres.

Run with the FreeCAD GUI Python console or FreeCADCmd:
    exec(open(".../ur5_m8_vision_target.py").read())

The generated STL is intended to print flat on the flange, with the bolt
pointing upward.  Do not use it for tightening, insertion, collision tests,
or any robot motion where it may contact equipment or people.
"""

import math
import os

import FreeCAD as App
import Mesh
import Part


# Robot-side interface: UR5 / UR5e, ISO 9409-1-50-4-M6.
FLANGE_OD = 76.0
FLANGE_THICKNESS = 8.0
M6_CLEARANCE = 6.8
BOLT_PCD = 50.0
BOLT_PATTERN_DEG = 45.0
LOCATING_PIN_CLEARANCE = 6.6

# Visual M8 bolt.  Set SHANK_LENGTH to match the height in your camera test.
M8_MAJOR_DIAMETER = 8.0
M8_ROOT_DIAMETER = 6.6
M8_PITCH = 1.25
SHANK_LENGTH = 28.0
TIP_CHAMFER_HEIGHT = 1.2
HEX_ACROSS_FLATS = 13.0
HEX_HEAD_HEIGHT = 5.3

# Wide load path between the printed flange and the narrow M8 mock-up.
REINFORCEMENT_BASE_DIAMETER = 36.0
REINFORCEMENT_TOP_DIAMETER = 18.0
REINFORCEMENT_HEIGHT = 8.0
ROOT_COLLAR_DIAMETER = 13.0
ROOT_COLLAR_HEIGHT = 5.0

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_NAME = "UR5_M8_Vision_Target"


def z_hole(x, y, diameter, z0, depth):
    return Part.makeCylinder(diameter / 2.0, depth, App.Vector(x, y, z0))


def hex_prism(across_flats, height, z0):
    """Regular hexagon whose opposing faces are `across_flats` apart."""
    circumradius = across_flats / math.sqrt(3.0)
    points = []
    # A flat is horizontal in the front view, as on a normal hex bolt.
    for i in range(6):
        a = math.radians(30.0 + i * 60.0)
        points.append(App.Vector(circumradius * math.cos(a),
                                 circumradius * math.sin(a), z0))
    points.append(points[0])
    return Part.Face(Part.makePolygon(points)).extrude(App.Vector(0, 0, height))


def cosmetic_thread(z0, length):
    """M8x1.25-like raised crests for camera appearance, not mating threads.

    Separate 0.70 mm-high tapered annular crests give a clear thread pattern
    in both RGB and depth images while avoiding a fragile true helical feature
    on an FDM prototype.
    """
    root_r = M8_ROOT_DIAMETER / 2.0
    major_r = M8_MAJOR_DIAMETER / 2.0
    straight_length = length - TIP_CHAMFER_HEIGHT
    result = Part.makeCylinder(root_r, straight_length,
                               App.Vector(0, 0, z0))
    # The free end is the threaded end of the simulated screw.
    result = result.fuse(Part.makeCone(
        root_r, root_r - 0.8, TIP_CHAMFER_HEIGHT,
        App.Vector(0, 0, z0 + straight_length)))
    crest_height = 0.70
    count = int(straight_length / M8_PITCH)
    for i in range(count):
        crest_z = z0 + i * M8_PITCH
        # Tapered annulus: 6.6 mm at the root, 8.0 mm at the crest.
        outer = Part.makeCone(root_r, major_r, crest_height,
                              App.Vector(0, 0, crest_z))
        inner = Part.makeCylinder(root_r, crest_height + 0.02,
                                  App.Vector(0, 0, crest_z - 0.01))
        result = result.fuse(outer.cut(inner))
    return result.removeSplitter()


def build_target():
    base = Part.makeCylinder(FLANGE_OD / 2.0, FLANGE_THICKNESS)
    bolt_r = BOLT_PCD / 2.0
    for index in range(4):
        angle = math.radians(BOLT_PATTERN_DEG + index * 90.0)
        x, y = bolt_r * math.cos(angle), bolt_r * math.sin(angle)
        base = base.cut(z_hole(x, y, M6_CLEARANCE, -1.0, FLANGE_THICKNESS + 2.0))

    # UR locating-pin relief.  Its orientation is not used for vision, but
    # retaining it makes repeated mounting orientation straightforward.
    base = base.cut(z_hole(bolt_r, 0, LOCATING_PIN_CLEARANCE,
                           -1.0, FLANGE_THICKNESS + 2.0))

    # Broad tapered hub prevents the printed part from concentrating all
    # bending stress at an unsupported 8 mm stem.
    reinforcement_z = FLANGE_THICKNESS
    reinforcement = Part.makeCone(
        REINFORCEMENT_BASE_DIAMETER / 2.0,
        REINFORCEMENT_TOP_DIAMETER / 2.0,
        REINFORCEMENT_HEIGHT,
        App.Vector(0, 0, reinforcement_z),
    )

    # Correct screw orientation: flange -> connection -> hex head -> threaded
    # shaft -> chamfered free end.
    head_z = reinforcement_z + REINFORCEMENT_HEIGHT
    head = hex_prism(HEX_ACROSS_FLATS, HEX_HEAD_HEIGHT, head_z)
    shank_z = head_z + HEX_HEAD_HEIGHT
    shank = cosmetic_thread(shank_z, SHANK_LENGTH)
    root_collar = Part.makeCone(
        ROOT_COLLAR_DIAMETER / 2.0,
        M8_MAJOR_DIAMETER / 2.0,
        ROOT_COLLAR_HEIGHT,
        App.Vector(0, 0, shank_z),
    )
    return (base.fuse(reinforcement).fuse(head).fuse(shank)
            .fuse(root_collar).removeSplitter())


try:
    doc = App.getDocument(DOC_NAME)
except Exception:
    doc = None
if doc is None:
    doc = App.newDocument(DOC_NAME)
old = doc.getObject("M8VisionTarget")
if old:
    doc.removeObject(old.Name)

target = doc.addObject("PartDesign::Feature", "M8VisionTarget")
target.Label = "UR5 M8 visual-detection target (non-load-bearing)"
target.Shape = build_target()
target.addProperty("App::PropertyString", "RobotInterface", "Design")
target.RobotInterface = "ISO 9409-1-50-4-M6; 4x dia 6.8 on PCD 50"
target.addProperty("App::PropertyString", "VisualFeature", "Design")
target.VisualFeature = "M8x1.25 cosmetic thread; 13 mm AF x 5.3 mm hex head"
target.addProperty("App::PropertyString", "Safety", "Design")
target.Safety = "Vision-only prototype; not an M8 mating thread or contact tool"
if target.ViewObject is not None:
    target.ViewObject.ShapeColor = (0.85, 0.62, 0.08)

doc.recompute()
os.makedirs(OUTPUT_DIR, exist_ok=True)
doc.saveAs(os.path.join(OUTPUT_DIR, "ur5_m8_vision_target.FCStd"))
Part.export([target], os.path.join(OUTPUT_DIR, "ur5_m8_vision_target.step"))
stl_path = os.path.join(OUTPUT_DIR, "ur5_m8_vision_target.stl")
Part.export([target], stl_path)
mesh = Mesh.Mesh(stl_path)
print("Created:", target.Label)
print("valid=", target.Shape.isValid(), "solids=", len(target.Shape.Solids),
      "facets=", mesh.CountFacets)
print("Height mm:", round(target.Shape.BoundBox.ZLength, 2))
print("Exported to:", OUTPUT_DIR)
