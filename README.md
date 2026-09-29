# Face Attendance Management System

A complete local web-based face attendance system inspired by the supplied UI.

## Features
- Admin login: `admin / admin`
- Dashboard with total students, today's present/absent and attendance percentage
- Student enrollment using laptop webcam
- Captures 8 face images
- OpenCV LBPH face recognition
- Automatic model training after enrollment
- Live attendance recognition from webcam
- Prevents duplicate attendance on the same day
- Students list with delete
- Attendance records with date/department filters
- Export attendance to CSV
- SQLite database
- Responsive dashboard UI

## Windows setup

### 1. Open terminal in this folder
```powershell
cd face_attendance_system
```

### 2. Create virtual environment
```powershell
py -3.13 -m venv venv
```

If `py -3.13` is not available, use:
```powershell
python -m venv venv
```

### 3. Activate
```powershell
.\venv\Scripts\activate
```

### 4. Install packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start
```powershell
python app.py
```

Open:
http://127.0.0.1:5000

Login:
- Username: `admin`
- Password: `admin`

## Important
Use `opencv-contrib-python`, not only `opencv-python`, because LBPH is inside `cv2.face`.

If you previously installed normal OpenCV and get:
`AttributeError: module 'cv2' has no attribute 'face'`

run:
```powershell
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

Then restart the terminal and run the app again.

## Camera permission
The browser will ask for webcam permission. Allow it.

## Database
SQLite is created automatically at:
`data/attendance.db`

Face images are stored at:
`data/faces/`

The trained model is stored at:
`data/trainer.yml`
