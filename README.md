# DICOM Slicer Tool

Small web app to upload DICOM (.dcm) files, select slices and extract them as PNG images, and download as a zip.

Features
- Supports uploading a single DICOM (.dcm) file
- Converts all image slices to PNG format and displays thumbnails
- Lets you select slices and download them as a ZIP archive
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
- All files (uploaded DICOMs and extracted PNGs) are stored in a temporary directory that is automatically cleaned up.
- When you click "Download Slices", the server builds an in-memory zip and returns it to the client.
- Files older than one hour are automatically cleaned up.
- Input validation and basic error handling are included (invalid files, invalid integer input).
