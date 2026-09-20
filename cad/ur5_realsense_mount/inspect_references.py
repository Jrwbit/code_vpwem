"""Inspect downloaded official Robotiq STEP references in FreeCADCmd."""

from pathlib import Path

import FreeCAD as App
import Import


root = Path("/home/wjr/code/vpwem/cad/ur5_realsense_mount/references")
for path in sorted(root.glob("*.step")):
    doc = App.newDocument(path.stem.replace("-", "_"))
    Import.insert(str(path), doc.Name)
    doc.recompute()
    shapes = []
    for obj in doc.Objects:
        try:
            if not obj.Shape.isNull():
                shapes.append(obj.Shape)
        except Exception:
            pass
    compound = shapes[0] if len(shapes) == 1 else __import__("Part").makeCompound(shapes)
    box = compound.BoundBox
    print(path.name)
    print("  objects:", len(doc.Objects), "shapes:", len(shapes))
    print("  envelope_mm:", round(box.XLength, 3), round(box.YLength, 3),
          round(box.ZLength, 3))
    print("  min:", round(box.XMin, 3), round(box.YMin, 3), round(box.ZMin, 3))
    print("  max:", round(box.XMax, 3), round(box.YMax, 3), round(box.ZMax, 3))
    App.closeDocument(doc.Name)
