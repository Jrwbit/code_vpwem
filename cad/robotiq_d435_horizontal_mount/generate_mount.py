"""Generate a flange-mounted horizontal Intel RealSense D435 bracket.

The printed part is installed in the ISO 9409-1-50-4-M6 interface between the
FT300-S tool face and the Hand-E GRP-CPL-062 coupling.  It is not a clamp.

Coordinates:
  * Z is the tool axis.  FT300-S is above +Z, Hand-E/camera are toward -Z.
  * The D435 90 mm long edge and its 45 mm M3 hole line run along X.
  * The camera is offset toward +Y.

Dimensions are millimetres.  Requires CadQuery 2.
"""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import cadquery as cq


HERE = Path(__file__).resolve().parent

# ISO 9409-1-50-4-M6 flange geometry.
FLANGE_OD = 75.0
FLANGE_THICKNESS = 5.0
BOLT_PCD = 50.0
BOLT_CLEARANCE = 6.7
BOLT_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)
CENTRE_BORE = 32.0                 # FT300-S drawing: nominal 31.5 H7
DOWEL_CLEARANCE = 6.4
DOWEL_RADIUS = 25.0

# Short horizontal camera plate.  D435 body is 90 x 25 x 25.05 mm.
CAMERA_CENTRE_Y = 55.0
CAMERA_PLATE_X = 100.0
CAMERA_PLATE_Y0 = 35.0
CAMERA_PLATE_Y1 = 73.0
CAMERA_PLATE_THICKNESS = FLANGE_THICKNESS
D435_HOLE_SPACING = 45.0
D435_M3_CLEARANCE = 3.4
D435_LENGTH = 90.0
D435_HEIGHT = 25.0
D435_DEPTH = 25.05

# Stiffening ribs are on the FT side, away from the camera body.
RIB_THICKNESS = 4.0
RIB_HEIGHT = 7.0


def through_cylinder(diameter: float, x: float, y: float) -> cq.Shape:
    """A cutter spanning the flange and ribs."""
    return (
        cq.Workplane("XY")
        .workplane(offset=-1.0)
        .center(x, y)
        .circle(diameter / 2.0)
        .extrude(FLANGE_THICKNESS + RIB_HEIGHT + 2.0)
        .val()
    )


def flange_mount() -> cq.Shape:
    # Full annular flange and the tangential horizontal camera plate.
    flange = cq.Workplane("XY").circle(FLANGE_OD / 2.0).extrude(FLANGE_THICKNESS).val()
    plate = (
        cq.Workplane("XY")
        .box(
            CAMERA_PLATE_X,
            CAMERA_PLATE_Y1 - CAMERA_PLATE_Y0,
            CAMERA_PLATE_THICKNESS,
            centered=(True, False, False),
        )
        .val()
        .translate((0.0, CAMERA_PLATE_Y0, 0.0))
    )
    # Wide roots connect the camera plate to the flange without sharp notches.
    web = (
        cq.Workplane("XY")
        .polyline([
            (-34.0, 27.0),
            (-46.0, CAMERA_PLATE_Y0),
            (46.0, CAMERA_PLATE_Y0),
            (34.0, 27.0),
        ])
        .close()
        .extrude(FLANGE_THICKNESS)
        .val()
    )
    shape = flange.fuse(plate).fuse(web)

    # Centre, four M6 clearances, and two alternate FT300-S dowel positions.
    # The production interfaces locate rotationally with the Ø6 dowel; adding
    # printed pilot fits would make the flange faces printer-dependent.
    # No counterbore/countersink: the screw heads remain seated in the existing
    # metal GRP-CPL-062 counterbores.
    shape = shape.cut(through_cylinder(CENTRE_BORE, 0.0, 0.0))
    for angle in BOLT_ANGLES_DEG:
        x = (BOLT_PCD / 2.0) * cos(radians(angle))
        y = (BOLT_PCD / 2.0) * sin(radians(angle))
        shape = shape.cut(through_cylinder(BOLT_CLEARANCE, x, y))
    for x in (-DOWEL_RADIUS, DOWEL_RADIUS):
        shape = shape.cut(through_cylinder(DOWEL_CLEARANCE, x, 0.0))

    # Two low triangular ribs resist camera pitch.  They sit outside the D435
    # screw line and do not form a protective cage.
    for x in (-36.0, 36.0):
        rib = (
            cq.Workplane("YZ")
            .workplane(offset=x - RIB_THICKNESS / 2.0)
            .polyline([
                (30.0, FLANGE_THICKNESS),
                (67.0, FLANGE_THICKNESS),
                (30.0, FLANGE_THICKNESS + RIB_HEIGHT),
            ])
            .close()
            .extrude(RIB_THICKNESS)
            .val()
        )
        shape = shape.fuse(rib)

    # D435 rear 1/4-20 hole is intentionally unused.  Use its two M3 holes;
    # leaving their heads proud prevents excessive thread insertion.
    for x in (-D435_HOLE_SPACING / 2.0, D435_HOLE_SPACING / 2.0):
        shape = shape.cut(through_cylinder(D435_M3_CLEARANCE, x, CAMERA_CENTRE_Y))

    return shape.clean()


def camera_envelope() -> cq.Shape:
    return (
        cq.Workplane("XY")
        .workplane(offset=-D435_DEPTH)
        .center(0.0, CAMERA_CENTRE_Y)
        .rect(D435_LENGTH, D435_HEIGHT)
        .extrude(D435_DEPTH)
        .val()
    )


def export_all() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    mount = flange_mount()
    camera = camera_envelope()

    assert mount.isValid(), "mount is not a valid B-rep"
    assert len(mount.Solids()) == 1, "mount must be one printable solid"
    assert mount.intersect(camera).Volume() < 1e-4, "mount intersects D435 envelope"

    stem = "ft300s_hande_d435_horizontal_flange_mount"
    cq.exporters.export(mount, str(HERE / f"{stem}.step"))
    cq.exporters.export(
        mount,
        str(HERE / f"{stem}.stl"),
        tolerance=0.04,
        angularTolerance=0.08,
    )

    assembly = cq.Assembly(name="FT300S_HandE_D435_horizontal_flange")
    assembly.add(mount, name="printed_flange_mount", color=cq.Color(0.15, 0.48, 0.86))
    assembly.add(camera, name="D435_envelope", color=cq.Color(0.08, 0.11, 0.16, 0.75))
    assembly.save(
        str(HERE / "ft300s_hande_d435_horizontal_flange_assembly.step"),
        exportType="STEP",
        mode="default",
    )

    box = mount.BoundingBox()
    camera_clearance = CAMERA_CENTRE_Y - D435_HEIGHT / 2.0 - FLANGE_OD / 2.0
    print("valid single solid:", mount.isValid(), len(mount.Solids()))
    print("mount envelope mm:", round(box.xlen, 2), round(box.ylen, 2), round(box.zlen, 2))
    print("D435-to-Ø75 coupling radial clearance mm:", round(camera_clearance, 2))
    print("M6 holes: Ø6.7 through, no head recess")
    print("D435 holes: Ø3.4 through, no head recess")


if __name__ == "__main__":
    export_all()
