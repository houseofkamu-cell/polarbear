# Golden Hour setup for DaVinci Resolve.
# Run inside Resolve: Workspace > Console > Py3, paste this whole file, press Enter.
# Before running: put GoldenHour_Rec709_33.cube and GoldenHour_plate_UHD.png in your Downloads folder.
import os, shutil, sys

LUT_NAME = "GoldenHour_Rec709_33.cube"
PLATE_NAME = "GoldenHour_plate_UHD.png"
src_dir = os.path.join(os.path.expanduser("~"), "Downloads")
lut_src = os.path.join(src_dir, LUT_NAME)
plate_src = os.path.join(src_dir, PLATE_NAME)

for p in (lut_src, plate_src):
    if not os.path.isfile(p):
        print("MISSING:", p, "- move it to Downloads and run again"); raise SystemExit

if sys.platform == "darwin":
    lut_dir = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
elif sys.platform.startswith("win"):
    lut_dir = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT"
else:
    lut_dir = "/opt/resolve/LUT"

lut_path = lut_src
try:
    dst = os.path.join(lut_dir, "GoldenHour")
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(lut_src, dst)
    lut_path = os.path.join(dst, LUT_NAME)
    print("LUT installed:", lut_path)
except Exception as e:
    print("Could not write to LUT folder (" + str(e) + "). Using LUT from Downloads instead.")

project = resolve.GetProjectManager().GetCurrentProject()
if not project:
    print("Open a project first."); raise SystemExit
project.RefreshLUTList()

pool = project.GetMediaPool()
imported = pool.ImportMedia([plate_src])
print("Plate imported to Media Pool:", bool(imported))

tl = project.GetCurrentTimeline()
if not tl:
    print("No timeline open. Plate is in the Media Pool; LUT not applied to any clips.")
    raise SystemExit

done, failed = 0, 0
for t in range(1, tl.GetTrackCount("video") + 1):
    for item in tl.GetItemListInTrack("video", t) or []:
        mpi = item.GetMediaPoolItem()
        if mpi and mpi.GetName() == PLATE_NAME:
            continue  # plate is already graded
        if item.SetLUT(1, lut_path):
            done += 1
        else:
            failed += 1
print("Golden Hour applied to node 1 on", done, "clips.", ("Failed on " + str(failed) + ".") if failed else "")
print("Next: drag the plate from the Media Pool onto the lowest video track under your footage.")
