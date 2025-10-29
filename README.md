# DICOM Slicer Tool

Small web app to upload a single DICOM (.dcm) file, extract every nth slice as PNG images, view them, and download as a zip.

Features
- Upload a single DICOM file (.dcm)
- Choose "every nth slice" to extract
- Converts selected slices to PNG and shows thumbnails
- Download extracted slices as a zip
- Simple, minimal UI using Jinja2 templates

Requirements
- Python 3.8+
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run

```powershell
# from project root
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Notes and behavior
- Uploaded DICOMs are saved to `uploads/` temporarily.
- Extracted PNGs are saved under `static/slices/<dicomname_timestamp>_slices/` so they can be displayed.
- When you click "Download Slices", the server builds an in-memory zip, returns it to the client, and deletes the generated PNG folder.
- The app attempts to cleanup uploaded DICOM files older than one hour on download.
- Input validation and basic error handling are included (invalid files, invalid integer input).

Adjust secret key in `app.py` before deploying publicly.