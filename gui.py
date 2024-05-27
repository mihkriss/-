import sys
import sqlite3
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
import cv2
from ultralytics import YOLO 
from xgboost import XGBClassifier
import numpy as np
from collections import deque
import folium
import random
from keras.models import load_model
import gpxpy


class VideoWidget(QWidget):
    data_processed = pyqtSignal(list, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_label = QLabel()
        self.video_label.setFixedSize(560, 590)
        self.video_label.setStyleSheet("background-color: #ffffff; border: none;")

        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.parent_window = parent
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.video_capture = cv2.VideoCapture('C:\\Users\\mikhe\\OneDrive\\Рабочий стол\\диплом\\ddd\\gr_kr.MOV')  
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # self.model = XGBClassifier()
        # self.model.load_model('xgboost.h5')
        self.model = load_model('lstm.h5')
        self.yolo_model = YOLO('yolov8n-pose.pt')
        self.last_frame = None
        self.video_writer = None
        self.video_file_path = None

    def start_video(self):
        if not self.timer.isActive():
            self.video_capture = cv2.VideoCapture('C:\\Users\\mikhe\\OneDrive\\Рабочий стол\\диплом\\ddd\\gr_kr.MOV') 
            self.timer.start(30)

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.video_capture.release()
            if self.last_frame is not None:
                self.process_frame(self.last_frame)  
                self.last_frame = None
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

    def process_frame(self, frame):
        results = self.yolo_model.predict(frame, conf=0.5, save=False)[0]
        keypoint = results.keypoints.xyn.cpu().numpy()
        box = results.boxes.xywhn.cpu().numpy()
        box = box.tolist()

        
        flattened_frame = keypoint.flatten()
        flattened_frame = np.concatenate([keypoint.flatten()])
        processed_frame = np.array([flattened_frame])
        prediction = self.model.predict(processed_frame)

        if prediction > 0.5:
            text = "Нет драки".format(prediction)
            cv2.putText(frame, text, (70, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

        else:
            text = "Драка".format(prediction)
            cv2.putText(frame, text, (70, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

        frame = results.plot()

        self.data_processed.emit(box, text)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.last_frame = frame
            self.process_frame(frame)
            
class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        icon_image = 'dron.png'

        layout = QVBoxLayout(self)
        self.webview = QWebEngineView()
        layout.addWidget(self.webview)

        сord = [55.855243, 37.480871]

        cord_ar = [56.097979, 35.879816]

        icon = folium.CustomIcon(
        icon_image,
        icon_size=(40, 40),
        icon_anchor=(30, 30))

        self.polygon_points_ar = [[56.100011, 35.873705], [56.102597, 35.877817], [56.097671, 35.884515], [56.094740, 35.881293]]
        self.polygon_points = [[55.855336, 37.480697], [55.855336, 37.4811039], [55.855336, 37.481103], [55.855133, 37.480705]]

        self.map = folium.Map(max_bounds=True, location=[55.855172, 37.4808916], zoom_start=50)
        self.tile = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri Satellite', overlay=False, control=True).add_to(self.map)

        self.marker = folium.Marker(location=[55.855172, 37.480891], popup="My Location", icon=icon, draggable=True)
        self.map.add_child(self.marker)

        self.track_coordinates = [self.marker.location]  
        self.track = folium.PolyLine(locations=self.track_coordinates, color='red', weight=2, opacity=0.75)
        self.map.add_child(self.track)

        self.fenceline_polygon = folium.Polygon(locations=self.polygon_points, color='blue', fill=True, fill_color='blue').add_to(self.map)

        self.coordinates_label = QLabel()
        self.coordinates_label.setStyleSheet("background-color: #ffffff; border: none; font-size: 14px;")
        self.coordinates_label.setFixedSize(450, 15)
        layout.addWidget(self.coordinates_label)

        self.status_label = QLabel("Статус: Неизвестен")  
        self.status_label.setStyleSheet("background-color: #ffffff; border: none; font-size: 14px;")
        self.status_label.setFixedSize(450, 15)
        layout.addWidget(self.status_label)

        self.map.save("map.html")
        self.webview.setHtml(open("map.html").read())

    def move_marker_random(self):
        new_location = [random.uniform(min(self.polygon_points)[0], max(self.polygon_points)[0]), 
                        random.uniform(min(self.polygon_points, key=lambda x: x[1])[1], 
                                       max(self.polygon_points, key=lambda x: x[1])[1])]

        if not ray_tracing(new_location[0], new_location[1], self.polygon_points):
            return
        
        inside_polygon = ray_tracing(new_location[0], new_location[1], self.polygon_points)
        
        if inside_polygon:
            self.status_label.setText("Статус: Внутри полигона")
        else:
            self.status_label.setText("Статус: Вне полигона")

        self.marker.location = new_location

        self.track_coordinates.append(new_location)
        
        self.track.locations = self.track_coordinates

        self.coordinates_label.setText(f"Координаты: {new_location[0]}, {new_location[1]}")

        self.map.save("map.html")
        self.webview.setHtml(open("map.html").read())

def ray_tracing(x, y, poly):
            n = len(poly)
            inside = False
            p2x = 0.0
            p2y = 0.0
            xints = 0.0
            p1x, p1y = poly[0]
            for i in range(n+1):
                p2x, p2y = poly[i % n]
                if y > min(p1y,p2y):
                    if y <= max(p1y,p2y):
                        if x <= max(p1x,p2x):
                            if p1y != p2y:
                                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сервис")
        self.setGeometry(100, 100, 2160, 1440)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout()
        self.central_widget.setLayout(layout)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")
        layout.addWidget(left_widget)

        left_top_layout = QVBoxLayout()
        left_top_layout.setContentsMargins(10, 10, 10, 10)
        left_top_widget = QWidget()
        left_top_widget.setLayout(left_top_layout)
        left_top_widget.setStyleSheet("background-color: #ffffff; border: 2px solid #999999; border-radius: 10px;")
        left_layout.addWidget(left_top_widget)

        left_bottom_layout = QVBoxLayout()
        left_bottom_layout.setContentsMargins(10, 10, 10, 10)
        left_bottom_layout.setSpacing(10)
        left_bottom_widget = QWidget()
        left_bottom_widget.setLayout(left_bottom_layout)
        left_bottom_widget.setStyleSheet("background-color: #ffffff; border: 2px solid #999999; border-radius: 10px;")
        left_layout.addWidget(left_bottom_widget)

        self.video_widget = VideoWidget()
        self.video_widget.data_processed.connect(self.create_database_table)
        left_top_layout.addWidget(self.video_widget)


        self.db_table = QTableWidget()
        left_bottom_layout.addWidget(self.db_table)
        self.create_database_table()
        self.db_table.setStyleSheet("background-color: #ffffff; border: none;")

        self.right_widget = QWidget()
        self.right_widget.setStyleSheet("background-color: #ffffff; border: 2px solid #999999; border-radius: 10px;")
        layout.addWidget(self.right_widget)

        self.map_widget = MapWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)
        self.right_layout.addWidget(self.map_widget)


        self.map_widget.setMinimumSize(700, 850)

        self.video_widget.setMinimumSize(560, 650)
        self.right_widget.setMinimumSize(800, 800)
        left_widget.setMinimumWidth(500)

        self.apply_button_style()

    def create_database_table(self, box = None, text = None):
            if box is None or text is None:
                return
            conn = sqlite3.connect('coordinats.db')
            
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS data
                     (id INTEGER PRIMARY KEY, label TEXT, box_x REAL, box_y REAL, box_w REAL, box_h REAL)''')

            c.execute("INSERT INTO data (label, box_x, box_y, box_w, box_h) VALUES (?, ?, ?, ?, ?)", ((text, box[0][0], box[0][1], box[0][2], box[0][3])))
            conn.commit()

            c.execute("SELECT * FROM data")
            data = c.fetchall()

            self.db_table.setColumnCount(len(data[0]))
            self.db_table.setRowCount(len(data))
            self.db_table.setHorizontalHeaderLabels(['ID', 'Метка', 'Коорд_X', 'Коорд_Y', 'Коопд_W', 'Коорд_H'])

            self.db_table.verticalHeader().setVisible(False)

            for i, row in enumerate(data):
                for j, col in enumerate(row):
                    item = QTableWidgetItem(str(col))
                    item.setBackground(QColor("#FFFFFF"))
                    self.db_table.setItem(i, j, item)

            conn.close()
            self.db_table.resizeColumnsToContents()

    def apply_button_style(self):
        button_style = """
            QPushButton {
                background-color: #00ADB5;
                color: #EEEEEE;
                border-radius: 5px;
                border: 2px solid #00ADB5;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #01878d;
                border: 2px solid #01878d;
            }
            QPushButton:pressed {
                background-color: #0f8085;
                border: 2px solid #0f8085;
            }
        """
        self.video_widget.start_button.setStyleSheet(button_style)
        self.video_widget.stop_button.setStyleSheet(button_style)
        self.show()

    def some_method(self):
        self.map_widget.move_marker_random()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Roboto")
    font.setPointSize(14)
    app.setFont(font)
    window = MainWindow()
    window.show()
    timer = QTimer()
    timer.timeout.connect(window.some_method)  
    timer.start(40000)
    sys.exit(app.exec())
