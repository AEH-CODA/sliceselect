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
- All files (uploaded DICOMs and extracted PNGs) are stored in a temporary directory that is automatically cleaned up.
- When you click "Download Slices", the server builds an in-memory zip and returns it to the client.
- Files older than one hour are automatically cleaned up.
- Input validation and basic error handling are included (invalid files, invalid integer input).