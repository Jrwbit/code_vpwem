# UR5 M8 visual-detection target

`ur5_m8_vision_target.py` generates a single printable fixture: a UR5/UR5e
four-hole M6 flange adapter with a centred M8-like visual bolt.  The corrected
stack is flange, reinforced tapered hub, hex head, threaded shaft, then the
free chamfered tip.  The bolt has an 8 mm shank, a 1.25 mm-pitch cosmetic
thread pattern, and a standard-looking 13 mm-across-flats hex head.

Run the script in FreeCAD to export `FCStd`, `STEP`, and `STL`.  Change
`SHANK_LENGTH` near the top of the script to set the exposed height.

Print it with the flange face on the bed.  PETG or PLA is sufficient for a
stationary camera-only validation; use four appropriate M6 robot-flange bolts.
This part is deliberately **not** a real M8 mating thread and must not be used
to tighten into the threaded hole, make contact, or carry a robot payload.
