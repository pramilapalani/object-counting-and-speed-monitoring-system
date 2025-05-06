# Intelligent Object Counting and Speed Monitoring System using YOLOv8

A real-time surveillance analytics tool to detect, track, count, and estimate the speed of objects using YOLOv8, with a web interface powered by Flask and deployable using Docker.

## 📌 Features
- Object detection & tracking with YOLOv8 + ByteTrack
- Global and region-wise object counting
- Speed estimation based on displacement over frames
- Web UI for video upload and live analytics
- CSV logs for analysis
- Docker-ready deployment

## 🧱 Project Structure
```
── app/
│   ├── static/
│   │   └── uploads/        # Uploaded/demo video files
│   ├── templates/
│   │   └── index.html      # Web UI template
│   ├── detection.py        # YOLOv8 detection, tracking, counting, speed estimation
│   └── app.py              # Flask web server
│
├── logs/
│   └── object_stats.csv    # Optional CSV logs (if enabled)
│
├── Dockerfile              # For containerization
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # YOLOv8 model weights (or link to download)
├── README.md               # Project overview and instructions
└── .gitignore              # To ignore files like pycache, .env, etc
```

## 🚀 Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/pramilapalani/object-counting-and-speed-monitoring-system.git
cd object-counting-and-speed-monitoring-system
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application
```bash
python app/app.py
```

### 5. Open the Web Interface
Go to your browser and open:
```
http://127.0.0.1:5000/
```

## 🐳 Docker Deployment
To deploy the project using Docker:
1. Build the Docker image:
   ```bash
   docker build -t intelligent-object-counter .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 intelligent-object-counter
   ```

Then, open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## 📊 Logs and Analytics
- Object statistics, such as counts and speed estimates, are saved in the `logs/object_stats.csv` file.
- Enable or configure logging in the `detection.py` file as needed.

## ⚙️ YOLOv8 Model
- The YOLOv8 model weights are included as `yolov8n.pt`.
- If the weights are not included, download them from the [official YOLOv8 repository](https://github.com/ultralytics/ultralytics) and place them in the root directory.

## 🛠️ Development Notes
- Edit the `app/templates/index.html` file to customize the web interface.
- Modify `detection.py` to adjust object detection, tracking, and speed estimation logic.

### 📽️ Demo
[Click to watch the demo video](app/static/output/output-demo.mp4)

## 📝 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## 📧 Contact
For questions or feedback, please contact [pramila](mailto:pramilapalani@gmail.com).
