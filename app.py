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
    # Two-step flow: upload file -> ask n. then same route with n provided to extract
    # If uploading file
    if 'dicom_file' in request.files and request.files['dicom_file'].filename != '':
        f = request.files['dicom_file']
        if not allowed_file(f.filename):
            flash('Please upload a file with .dcm extension')
            return redirect(url_for('index'))
        filename = os.path.basename(f.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        save_name = f"{os.path.splitext(filename)[0]}_{timestamp}.dcm"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        f.save(save_path)
        # show form to select n
        return render_template('result.html', stage='choose_n', uploaded_filename=save_name)

    # If user submitted the nth value or requested to show all slices
    uploaded = request.form.get('uploaded_filename')
    n_value = request.form.get('n')
    action = request.form.get('action')
    if not uploaded:
        flash('No uploaded file found. Please upload a DICOM file first.')
        return redirect(url_for('index'))
    upload_path = os.path.join(UPLOAD_FOLDER, uploaded)
    if not os.path.exists(upload_path):
        flash('Uploaded file not found on server. Try uploading again.')
        return redirect(url_for('index'))

    # validate n
    # If user clicked "Show all slices", we will generate all frames and show them
    if action == 'show_all':
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

        # prepare output folder
        base = os.path.splitext(uploaded)[0]
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

        # Create URLs using the serve_image route instead of static files
        thumbnails = [url_for('serve_image', folder=folder_name, filename=fn) for fn in saved_files]
        return render_template('result.html', stage='show_results', folder_name=folder_name, thumbnails=thumbnails, filenames=saved_files)

    # Otherwise validate n
    try:
        n = int(n_value)
        if n <= 0:
            raise ValueError()
    except Exception:
        flash('Please provide a valid positive integer for nth slice')
        return render_template('result.html', stage='choose_n', uploaded_filename=uploaded)

    # Load DICOM
    try:
        arr = read_dicom_pixel_array(upload_path)
    except pydicom.errors.InvalidDicomError:
        flash('Uploaded file is not a valid DICOM file.')
        # cleanup uploaded file
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Failed to read DICOM: {e}')
        return redirect(url_for('index'))

    # Determine frames
    if arr.ndim == 2:
        total = 1
    else:
        total = arr.shape[0]

    # select every nth slice starting from first
    indices = list(range(0, total, n))
    if len(indices) == 0:
        flash('No slices selected with this frequency. Try a smaller n.')
        return render_template('result.html', stage='choose_n', uploaded_filename=uploaded)

    # Prepare output folder name
    base = os.path.splitext(uploaded)[0]
    folder_name = f"{base}_slices"
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)
    # If exists from previous run, clear it
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # If multi-frame, pick every nth frame and save
    to_save = []
    if arr.ndim == 2:
        selected = [arr]
    else:
        selected = [arr[i] for i in indices]

    saved_files = []
    for idx, a in zip(indices, selected):
        # normalize and save each as PNG
        img_arr = normalize_to_uint8(a)
        im = Image.fromarray(img_arr)
        if im.mode != 'RGB':
            im = im.convert('L').convert('RGB')
        fname = f"slice_{idx+1:04d}.png"
        im.save(os.path.join(out_dir, fname))
        saved_files.append(fname)

    # Keep a reference to cleanup later by returning folder key
    # Show thumbnails
    thumbnails = []
    for fn in saved_files:
        # Create route-based URLs that will serve from temp directory
        thumbnails.append(url_for('serve_image', folder=folder_name, filename=fn))

    # Render results page with thumbnails and download link
    return render_template('result.html', stage='show_results', folder_name=folder_name, thumbnails=thumbnails, filenames=saved_files)


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