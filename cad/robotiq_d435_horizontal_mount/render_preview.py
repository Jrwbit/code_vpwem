"""Render an annotated geometry preview of the flange-mounted D435 bracket."""

from pathlib import Path

import vtk


HERE = Path(__file__).resolve().parent
PNG = HERE / "ft300s_hande_d435_horizontal_flange_preview.png"


def add_stl(renderer, filename, color):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(HERE / filename))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetInterpolationToPhong()
    renderer.AddActor(actor)


def add_cylinder(renderer, radius, height, z, color, opacity):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(128)
    source.SetCenter(0.0, 0.0, z + height / 2.0)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.RotateX(90.0)
    # vtkCylinderSource is Y-axis by default.  Rotation changes the location;
    # restore it explicitly in world coordinates.
    actor.SetPosition(0.0, z + height / 2.0, -(z + height / 2.0))
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)


renderer = vtk.vtkRenderer()
renderer.SetBackground(0.98, 0.98, 0.98)
add_stl(renderer, "ft300s_hande_d435_horizontal_flange_mount.stl", (0.10, 0.43, 0.86))

# Simplified adjoining hardware: FT300-S above the printed flange, Hand-E
# coupling and gripper below it.  They show the sandwich location, not detail.
add_cylinder(renderer, 37.5, 13.0, 5.0, (0.36, 0.36, 0.38), 0.72)
add_cylinder(renderer, 37.5, 16.9, -19.7, (0.48, 0.48, 0.50), 0.58)
add_cylinder(renderer, 37.5, 36.0, -55.7, (0.28, 0.29, 0.32), 0.42)

# D435: its long edge is horizontal along X, with the lens face toward -Z.
camera = vtk.vtkCubeSource()
camera.SetXLength(90.0)
camera.SetYLength(25.0)
camera.SetZLength(25.05)
camera.SetCenter(0.0, 55.0, -12.525)
camera_mapper = vtk.vtkPolyDataMapper()
camera_mapper.SetInputConnection(camera.GetOutputPort())
camera_actor = vtk.vtkActor()
camera_actor.SetMapper(camera_mapper)
camera_actor.GetProperty().SetColor(0.06, 0.09, 0.14)
camera_actor.GetProperty().SetOpacity(0.93)
renderer.AddActor(camera_actor)

window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(1200, 900)
window.AddRenderer(renderer)
view = renderer.GetActiveCamera()
view.SetPosition(170.0, 205.0, 140.0)
view.SetFocalPoint(0.0, 25.0, -10.0)
view.SetViewUp(0.0, 0.0, 1.0)
renderer.ResetCameraClippingRange()
window.Render()

capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.SetScale(2)
capture.SetInputBufferTypeToRGBA()
capture.ReadFrontBufferOff()
capture.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(str(PNG))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
print(PNG)
