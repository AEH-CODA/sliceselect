import os
import io
import zipfile
import shutil
from datetime import datetime
import tempfile
import atexit

from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.dataset import FileDataset
import numpy as np
from PIL import Image

# Create temporary directories that will be cleaned up when the server stops
TEMP_ROOT = tempfile.mkdtemp(prefix='dicom_slicer_')
UPLOAD_FOLDER = os.path.join(TEMP_ROOT, 'uploads')
OUTPUT_ROOT = os.path.join(TEMP_ROOT, 'slices')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Register cleanup of temp directories when server exits
@atexit.register
def cleanup_temp_files():
    try:
        shutil.rmtree(TEMP_ROOT)
    except Exception:
        pass

app = Flask(__name__)
app.secret_key = 'change-me-to-a-random-secret'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit
app.config['STATIC_FOLDER'] = None  # Disable static folder


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.lower().endswith('.dcm')


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= np.min(arr)
    maxv = np.max(arr)
    if maxv > 0:
        arr /= maxv
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def read_dicom_pixel_array(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    # Try to apply VOI LUT when possible for correct contrast
    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        arr = ds.pixel_array
    # If monochrome with PhotometricInterpretation=='MONOCHROME1', invert
    try:
        pi = ds.get('PhotometricInterpretation', '').upper()
        if pi == 'MONOCHROME1':
            arr = np.max(arr) - arr
    except Exception:
        pass
    return arr


def save_slices_as_pngs(arr: np.ndarray, out_dir: str, prefix: str = 'slice') -> list:
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []
    # If single 2D array, make it 3D with one frame
    if arr.ndim == 2:
        frames = 1
        arrs = [arr]
    elif arr.ndim == 3:
        # assume (frames, rows, cols)
        frames = arr.shape[0]
        arrs = [arr[i] for i in range(frames)]
    else:
        # handle unexpected shapes by flattening outer dims
        frames = arr.shape[0]
        arrs = [arr[i] for i in range(frames)]

    for i, a in enumerate(arrs):
        img_arr = normalize_to_uint8(a)
        im = Image.fromarray(img_arr)
        # convert to RGB for broader browser compatibility
        if im.mode != 'RGB':
            im = im.convert('L').convert('RGB')
        filename = f"{prefix}_{i+1:04d}.png"
        full = os.path.join(out_dir, filename)
        im.save(full)
        saved_files.append(filename)
    return saved_files


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'dicom_file' not in request.files or request.files['dicom_file'].filename == '':
        flash('No file uploaded')
        return redirect(url_for('index'))
        
    f = request.files['dicom_file']
    if not allowed_file(f.filename):
        flash('Please upload a file with .dcm extension')
        return redirect(url_for('index'))

    # Save uploaded file
    filename = os.path.basename(f.filename)
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    save_name = f"{os.path.splitext(filename)[0]}_{timestamp}.dcm"
    upload_path = os.path.join(UPLOAD_FOLDER, save_name)
    f.save(upload_path)

    try:
        arr = read_dicom_pixel_array(upload_path)
    except pydicom.errors.InvalidDicomError:
        flash('Uploaded file is not a valid DICOM file.')
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Failed to read DICOM: {e}')
        return redirect(url_for('index'))

    # Prepare output folder
    base = os.path.splitext(save_name)[0]
    folder_name = f"{base}_slices"
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save all frames
    if arr.ndim == 2:
        frames = [arr]
        indices = [0]
    else:
        frames = [arr[i] for i in range(arr.shape[0])]
        indices = list(range(len(frames)))

    saved_files = []
    for idx, a in zip(indices, frames):
        img_arr = normalize_to_uint8(a)
        im = Image.fromarray(img_arr)
        if im.mode != 'RGB':
            im = im.convert('L').convert('RGB')
        fname = f"slice_{idx+1:04d}.png"
        im.save(os.path.join(out_dir, fname))
        saved_files.append(fname)

    # Create URLs using the serve_image route
    thumbnails = [url_for('serve_image', folder=folder_name, filename=fn) for fn in saved_files]

    # Pass upload_path to result template so we can access DICOM tags later
    return render_template('result.html', stage='show_results', folder_name=folder_name, thumbnails=thumbnails, filenames=saved_files, upload_path=upload_path)


@app.route('/image/<folder>/<filename>')
def serve_image(folder, filename):
    """Serve images from temporary directory"""
    # Sanitize inputs to prevent directory traversal
    folder = os.path.basename(folder)
    filename = os.path.basename(filename)
    path = os.path.join(OUTPUT_ROOT, folder, filename)
    if not os.path.exists(path):
        return 'Image not found', 404
    return send_file(path, mimetype='image/png')


def copy_dicom_tags(source_ds: pydicom.Dataset, target_ds: FileDataset) -> None:
    """Copy important DICOM tags from source to target dataset"""
    # List of important tags to copy
    important_tags = [
        'PatientName', 'PatientID', 'PatientAge', 'PatientBirthDate', 'PatientSex',
        'StudyInstanceUID', 'SeriesInstanceUID', 'SeriesNumber',
        'PixelSpacing', 'SliceLocation', 'SliceThickness',
        'EchoTime', 'RepetitionTime', 'FlipAngle',
        'AcquisitionDate', 'AcquisitionTime',
        'Modality', 'Manufacturer', 'ManufacturerModelName',
        'InstitutionName', 'ReferringPhysicianName',
        'StudyDate', 'SeriesDate',
        'ImagePositionPatient', 'ImageOrientationPatient',
        'BitsAllocated', 'BitsStored', 'HighBit',
        'PhotometricInterpretation', 'SamplesPerPixel',
        'RescaleSlope', 'RescaleIntercept', 'WindowCenter', 'WindowWidth'
    ]
    
    for tag_name in important_tags:
        try:
            if hasattr(source_ds, tag_name):
                value = getattr(source_ds, tag_name)
                if value is not None:  # Only copy if value exists
                    setattr(target_ds, tag_name, value)
        except Exception:
            # Skip tags that can't be copied - continue gracefully
            pass


def create_dicom_from_slices(upload_path: str, selected_indices: list, pixel_array: np.ndarray) -> bytes:
    """Create a DICOM file from selected slices with preserved tags"""
    # Read original DICOM to get tags
    source_ds = pydicom.dcmread(upload_path)
    
    # Extract selected slices from pixel array
    if pixel_array.ndim == 2:
        selected_pixel_data = pixel_array
    else:
        # Multi-frame DICOM - select specific frames
        selected_frames = [pixel_array[i] for i in selected_indices]
        selected_pixel_data = np.stack(selected_frames, axis=0)
    
    # Ensure pixel data is in appropriate integer format
    # Convert float to uint16 if needed
    if selected_pixel_data.dtype == np.float32 or selected_pixel_data.dtype == np.float64:
        # Normalize to appropriate range
        pixel_min = np.min(selected_pixel_data)
        pixel_max = np.max(selected_pixel_data)
        if pixel_max > pixel_min:
            selected_pixel_data = ((selected_pixel_data - pixel_min) / (pixel_max - pixel_min) * 65535).astype(np.uint16)
        else:
            selected_pixel_data = selected_pixel_data.astype(np.uint16)
    elif selected_pixel_data.dtype not in [np.uint16, np.uint8, np.int16]:
        selected_pixel_data = selected_pixel_data.astype(np.uint16)
    
    # Create new FileDataset - filename is a positional argument, not keyword
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    new_ds = FileDataset("output.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Copy important tags
    copy_dicom_tags(source_ds, new_ds)
    
    # Set pixel data
    new_ds.PixelData = selected_pixel_data.tobytes()
    
    # Ensure pixel array shape is set correctly
    if selected_pixel_data.ndim == 3:
        new_ds.NumberOfFrames = selected_pixel_data.shape[0]
        new_ds.Rows = selected_pixel_data.shape[1]
        new_ds.Columns = selected_pixel_data.shape[2]
    else:
        new_ds.Rows = selected_pixel_data.shape[0]
        new_ds.Columns = selected_pixel_data.shape[1]
    
    # Ensure required tags are set with safe defaults
    try:
        if not hasattr(new_ds, 'PatientName') or new_ds.PatientName is None:
            new_ds.PatientName = 'Anonymous'
    except Exception:
        new_ds.PatientName = 'Anonymous'
    
    try:
        if not hasattr(new_ds, 'PatientID') or new_ds.PatientID is None:
            new_ds.PatientID = 'Unknown'
    except Exception:
        new_ds.PatientID = 'Unknown'
    
    try:
        if not hasattr(new_ds, 'Modality') or new_ds.Modality is None:
            new_ds.Modality = 'OT'
    except Exception:
        new_ds.Modality = 'OT'
    
    try:
        if not hasattr(new_ds, 'SamplesPerPixel') or new_ds.SamplesPerPixel is None:
            new_ds.SamplesPerPixel = 1
    except Exception:
        new_ds.SamplesPerPixel = 1
    
    try:
        if not hasattr(new_ds, 'PhotometricInterpretation') or new_ds.PhotometricInterpretation is None:
            new_ds.PhotometricInterpretation = 'MONOCHROME2'
    except Exception:
        new_ds.PhotometricInterpretation = 'MONOCHROME2'
    
    try:
        if not hasattr(new_ds, 'BitsAllocated') or new_ds.BitsAllocated is None:
            new_ds.BitsAllocated = 16
    except Exception:
        new_ds.BitsAllocated = 16
    
    try:
        if not hasattr(new_ds, 'BitsStored') or new_ds.BitsStored is None:
            new_ds.BitsStored = 16
    except Exception:
        new_ds.BitsStored = 16
    
    try:
        if not hasattr(new_ds, 'HighBit') or new_ds.HighBit is None:
            new_ds.HighBit = 15
    except Exception:
        new_ds.HighBit = 15
    
    try:
        if not hasattr(new_ds, 'PixelRepresentation') or new_ds.PixelRepresentation is None:
            new_ds.PixelRepresentation = 0
    except Exception:
        new_ds.PixelRepresentation = 0
    
    # Write to bytes
    output = io.BytesIO()
    new_ds.save_as(output)
    output.seek(0)
    return output.getvalue()

@app.route('/download')
def download():
    folder = request.args.get('folder')
    if not folder:
        flash('No folder specified for download')
        return redirect(url_for('index'))
    out_dir = os.path.join(OUTPUT_ROOT, folder)
    if not os.path.exists(out_dir):
        flash('Requested folder not found')
        return redirect(url_for('index'))

    # Create zip in memory
    zip_bytes = io.BytesIO()
    zf = zipfile.ZipFile(zip_bytes, mode='w', compression=zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(out_dir):
        for file in files:
            full = os.path.join(root, file)
            arcname = os.path.join(folder, file)
            zf.write(full, arcname)
    zf.close()
    zip_bytes.seek(0)

    # Cleanup images and uploaded dicoms associated with this folder
    try:
        shutil.rmtree(out_dir)
    except Exception:
        pass
    # Clean up old files
    cleanup_old_files()

    return send_file(zip_bytes, mimetype='application/zip', as_attachment=True, download_name=f'{folder}.zip')


@app.route('/download_selected', methods=['POST'])
def download_selected():
    folder = request.form.get('folder')
    selected = request.form.getlist('selected')
    if not folder or not selected:
        flash('No files selected for download')
        return redirect(url_for('index'))
    out_dir = os.path.join(OUTPUT_ROOT, folder)
    if not os.path.exists(out_dir):
        flash('Requested folder not found')
        return redirect(url_for('index'))

    zip_bytes = io.BytesIO()
    zf = zipfile.ZipFile(zip_bytes, mode='w', compression=zipfile.ZIP_DEFLATED)
    for fn in selected:
        safe_name = os.path.basename(fn)
        full = os.path.join(out_dir, safe_name)
        if os.path.exists(full):
            arcname = os.path.join(folder, safe_name)
            zf.write(full, arcname)
    zf.close()
    zip_bytes.seek(0)

    # If desired, remove the created images folder after download
    try:
        shutil.rmtree(out_dir)
    except Exception:
        pass

    # cleanup old uploads as before
    try:
        now = datetime.utcnow()
        for fn in os.listdir(UPLOAD_FOLDER):
            p = os.path.join(UPLOAD_FOLDER, fn)
            if os.path.isfile(p):
                mtime = datetime.utcfromtimestamp(os.path.getmtime(p))
                if (now - mtime).total_seconds() > 3600:  # 1 hour
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception:
        pass

    return send_file(zip_bytes, mimetype='application/zip', as_attachment=True, download_name=f'{folder}.zip')


@app.route('/download_selected_dicom', methods=['POST'])
def download_selected_dicom():
    """Download selected slices as a DICOM file with preserved tags"""
    folder = request.form.get('folder')
    selected = request.form.getlist('selected')
    upload_path = request.form.get('upload_path')
    
    if not folder or not selected or not upload_path:
        flash('Missing required data for DICOM download')
        return redirect(url_for('index'))
    
    if not os.path.exists(upload_path):
        flash('Original DICOM file not found')
        return redirect(url_for('index'))
    
    try:
        # Read original DICOM to get pixel array and tags
        pixel_array = read_dicom_pixel_array(upload_path)
        
        # Convert selected filenames to indices
        # Filenames are like: slice_0001.png, slice_0002.png, etc.
        selected_indices = []
        for fn in selected:
            safe_name = os.path.basename(fn)
            # Extract the number from slice_XXXX.png
            try:
                num_str = safe_name.split('_')[1].split('.')[0]
                idx = int(num_str) - 1  # Convert to 0-based index
                selected_indices.append(idx)
            except Exception:
                continue
        
        if not selected_indices:
            flash('Could not parse selected slice indices')
            return redirect(url_for('index'))
        
        # Sort indices to maintain order
        selected_indices.sort()
        
        # Create DICOM from selected slices
        dicom_bytes = create_dicom_from_slices(upload_path, selected_indices, pixel_array)
        
        # Create filename
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        dicom_filename = f"dicom_slices_{timestamp}.dcm"
        
        # Return as download
        return send_file(
            io.BytesIO(dicom_bytes),
            mimetype='application/dicom',
            as_attachment=True,
            download_name=dicom_filename
        )
    
    except Exception as e:
        flash(f'Error creating DICOM file: {str(e)}')
        return redirect(url_for('index'))
    
    finally:
        # Cleanup after download
        out_dir = os.path.join(OUTPUT_ROOT, folder)
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass
        cleanup_old_files()


def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        now = datetime.utcnow()
        # Clean old uploads
        for fn in os.listdir(UPLOAD_FOLDER):
            p = os.path.join(UPLOAD_FOLDER, fn)
            if os.path.isfile(p):
                mtime = datetime.utcfromtimestamp(os.path.getmtime(p))
                if (now - mtime).total_seconds() > 3600:  # 1 hour
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        # Clean old slice folders
        for fn in os.listdir(OUTPUT_ROOT):
            p = os.path.join(OUTPUT_ROOT, fn)
            if os.path.isdir(p):
                mtime = datetime.utcfromtimestamp(os.path.getmtime(p))
                if (now - mtime).total_seconds() > 3600:  # 1 hour
                    try:
                        shutil.rmtree(p)
                    except Exception:
                        pass
    except Exception:
        pass


if __name__ == '__main__':
    app.run(debug=True)