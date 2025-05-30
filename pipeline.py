import sys
import joblib
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QStackedWidget, QMessageBox, QGridLayout,
)
from PyQt6.QtGui import QFont, QDoubleValidator, QIntValidator, QColor
from PyQt6.QtCore import Qt


class ModelPredictor:
    def __init__(self):
        # Load the trained model
        self.model = joblib.load('xgboost_final_model.pkl')

    @staticmethod
    def preprocess_input(input_data):
        # If the input is a CSV file path, convert it to a DataFrame
        if isinstance(input_data, str):
            input_data = pd.read_csv(input_data)

        # If the input is a dictionary, convert it to a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Remove unnecessary columns such as 'Label'
        if 'Label' in input_data.columns:
            input_data = input_data.drop(columns=['Label'])

        return input_data

    def predict(self, input_data):
        # Preprocess the input data
        X = self.preprocess_input(input_data)

        # Perform predictions using the model
        predictions = self.model.predict(X)
        return predictions


# Instantiate the model for direct usage
Model = ModelPredictor()


class StyledButton(QPushButton):
    def __init__(self, text, color='primary'):
        super().__init__(text)
        self.setFont(QFont('Arial', 12, QFont.Weight.Medium))

        # Define color schemes for different button states
        color_schemes = {
            'primary': {
                'background': '#1A73E8',
                'hover': '#185ABC',
                'text': 'white'
            },
            'secondary': {
                'background': '#34A853',
                'hover': '#2B8D46',
                'text': 'white'
            },
            'danger': {
                'background': '#EA4335',
                'hover': '#C5221F',
                'text': 'white'
            }
        }

        scheme = color_schemes.get(color, color_schemes['primary'])

        # Set the style sheet for the button appearance
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {scheme['background']};
                color: {scheme['text']};
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {scheme['hover']};
            }}
        """)


class ManualInputPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        # Set up the UI for manual input page
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        # Create a header label
        header = QLabel(" پیش بینی دستی")
        header.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Define input fields and associated validators
        input_fields = [
            ("COMP_NAME", QDoubleValidator()),
            ("Vel, Rms (RMS)", QDoubleValidator()),
            ("Acc, Rms (RMS)", QDoubleValidator()),
            ("Crest (RMS)", QDoubleValidator()),
            ("Kurt (RMS)", QDoubleValidator()),
            ("Vel, Peak (RMS)", QDoubleValidator()),
            ("Vel, Peak to peak (RMS)", QDoubleValidator()),
            ("MP_LOC", QIntValidator()),
        ]

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(15)
        grid_layout.setVerticalSpacing(10)

        self.input_widgets = {}
        for row, (label_text, validator) in enumerate(input_fields):
            label = QLabel(label_text)
            label.setFont(QFont('Arial', 12))

            line_edit = QLineEdit()
            if validator:
                line_edit.setValidator(validator)
            line_edit.setStyleSheet("""
                QLineEdit {
                    padding: 10px;
                    border: 1px solid #DADCE0;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border-color: #1A73E8;
                }
            """)

            grid_layout.addWidget(label, row, 0)
            grid_layout.addWidget(line_edit, row, 1)

            self.input_widgets[label_text] = line_edit

        layout.addLayout(grid_layout)

        # Create predict and back buttons
        button_layout = QHBoxLayout()
        predict_btn = StyledButton("پیش بینی", 'primary')
        predict_btn.clicked.connect(self._make_prediction)
        back_btn = StyledButton("بازگشت", 'secondary')
        back_btn.clicked.connect(self.parent.show_home)

        button_layout.addWidget(predict_btn)
        button_layout.addWidget(back_btn)
        layout.addLayout(button_layout)

        # Label to show prediction results
        self.result_label = QLabel()
        self.result_label.setFont(QFont('Arial', 14))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        layout.addStretch()

    def _make_prediction(self):
        # Handle the prediction logic when the predict button is clicked
        try:
            # Collect input values from widgets
            input_dict = {
                field_name: widget.text() for field_name, widget in self.input_widgets.items()
            }

            # Convert input values to numeric types
            numeric_input = {k: float(v) for k, v in input_dict.items() if v.strip()}

            # Raise an error if any required field is empty
            if len(numeric_input) != len(self.input_widgets):
                raise ValueError("برخی از فیلدها خالی هستند.")

            # Run the prediction with the prepared input data
            prediction = Model.predict(numeric_input)
            self.result_label.setText(f"نتیجه پیش بینی: {prediction}")
            self.result_label.setStyleSheet("color: #34A853;")
        except ValueError:
            # Show a warning message if inputs are not correctly filled
            QMessageBox.warning(self, "خطا در ورودی‌ها", "لطفا همه‌ی فیلدها را به‌درستی پر کنید.")
        except Exception as e:
            # Show a critical error message for unexpected errors
            QMessageBox.critical(self, "خطا", f"مشکلی پیش آمد: {e}")


class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        # Setup UI for the home page
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        # Create a header label for the home page
        header = QLabel("ابزار تست مدل هوش مصنوعی")
        header.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Layout for main navigation buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        # Button to go to the manual input page
        manual_btn = StyledButton("ورود دستی", 'primary')
        manual_btn.clicked.connect(self.parent.show_manual_input)
        buttons_layout.addWidget(manual_btn)

        # Button to go to the CSV upload page
        csv_btn = StyledButton("آپلود فایل CSV", 'secondary')
        csv_btn.clicked.connect(self.parent.show_csv_upload)
        buttons_layout.addWidget(csv_btn)

        layout.addLayout(buttons_layout)
        layout.addStretch()


class CSVUploadPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        # Setup UI for the CSV upload page

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)

        # Page title
        header = QLabel("بارگذاری فایل CSV")
        header.setFont(QFont('IranSans', 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(header)

        # Table widget to show results and predictions
        self.results_table = QTableWidget()
        self.results_table.setFixedHeight(500)  # Set a larger height
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f2f2f2;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                border: 1px solid #ddd;
            }
        """)
        self.results_table.setAlternatingRowColors(True)
        content_layout.addWidget(self.results_table)

        # Label to display the accuracy after predictions
        self.accuracy_label = QLabel("نتیجه پیش‌بینی: -")
        self.accuracy_label.setFont(QFont('IranSans', 14))
        self.accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.accuracy_label)

        content_layout.addStretch(1)

        # Buttons for CSV upload and navigation
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        upload_btn = QPushButton("انتخاب فایل CSV")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        upload_btn.clicked.connect(self._upload_csv)
        button_layout.addWidget(upload_btn)

        back_btn = QPushButton("بازگشت")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        back_btn.clicked.connect(self.parent.show_home)
        button_layout.addWidget(back_btn)

        content_layout.addLayout(button_layout)
        main_layout.addLayout(content_layout)

    def _upload_csv(self):
        # Open a file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل CSV", "", "فایل‌های CSV (*.csv)")
        if file_path:
            try:
                self._process_csv(file_path)
            except Exception as e:
                QMessageBox.critical(self, "خطای بارگذاری CSV", str(e))

    def _process_csv(self, file_path):
        # Process the selected CSV file and display results in the table
        try:
            df = pd.read_csv(file_path)

            # Required columns for prediction
            columns_for_prediction = [
                "COMP_NAME",
                "Vel, Rms (RMS)",
                "Acc, Rms (RMS)",
                "Crest (RMS)",
                "Kurt (RMS)",
                "Vel, Peak (RMS)",
                "Vel, Peak to peak (RMS)",
                "MP_LOC"
            ]

            # Check if all required columns exist
            missing_columns = set(columns_for_prediction) - set(df.columns)
            if missing_columns:
                raise KeyError(f"ستون‌های زیر وجود ندارند: {', '.join(missing_columns)}")

            # Make predictions and add them to the DataFrame
            predictions = Model.predict(df[columns_for_prediction])
            df['Prediction'] = predictions

            # Calculate accuracy if 'Label' column exists
            correct_predictions = 0
            has_label = 'Label' in df.columns

            # Set up the table widget to show the data
            self.results_table.setRowCount(len(df))
            self.results_table.setColumnCount(len(df.columns))
            self.results_table.setHorizontalHeaderLabels(df.columns)

            # Populate the table with DataFrame values
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    value = str(df.iloc[row, col])
                    item = QTableWidgetItem(value)

                    # بررسی پیش‌بینی برای رنگ‌بندی
                    # Color-code cells based on prediction correctness if Label exists
                    if df.columns[col] == 'Prediction' and has_label:
                        prediction = df.iloc[row, col]
                        actual_label = df.iloc[row]['Label']

                        if prediction == actual_label:
                            item.setBackground(QColor(200, 255, 200))  # Green for correct prediction
                            correct_predictions += 1
                        else:
                            item.setBackground(QColor(255, 200, 200))  # Red for incorrect prediction
                    elif df.columns[col] == 'Prediction':
                        # If there's no 'Label', just color it gray
                        item.setBackground(QColor(220, 220, 220))  # Gray when no actual label provided

                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.results_table.setItem(row, col, item)

            # Display accuracy if label is present
            if has_label:
                accuracy = (correct_predictions / len(df)) * 100
                self.accuracy_label.setText(f"محاسبه برای {len(df)} سطر انجام شد. دقت: {accuracy:.2f}%")
            else:
                self.accuracy_label.setText(f"محاسبه برای {len(df)} سطر انجام شد. برچسبی برای محاسبه دقت موجود نیست")
                # New Lines
                df.rename(columns={"Prediction": "Label"}, inplace=True)
                df.to_csv("Result.csv", index=False)

            self.results_table.resizeColumnsToContents()

        except KeyError as e:
            # Show a warning if required columns are missing
            QMessageBox.warning(self, "خطا در ستون‌ها", f"ستون موردنیاز یافت نشد: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ابزار تست مدل هوش مصنوعی")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #F8F9FA;")

        # Central Widget and Stacked Pages
        # Create a central widget and a stacked widget to hold multiple pages
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Create Pages
        # Instantiate the pages (home, manual input, CSV upload)
        self.home_page = HomePage(self)
        self.manual_input_page = ManualInputPage(self)
        self.csv_upload_page = CSVUploadPage(self)

        # Add Pages to Stacked Widget
        # Add the instantiated pages to the stacked widget
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.manual_input_page)
        self.stacked_widget.addWidget(self.csv_upload_page)

        self.setCentralWidget(central_widget)

    def show_home(self):
        # Switch to the home page
        self.stacked_widget.setCurrentWidget(self.home_page)

    def show_manual_input(self):
        # Switch to the manual input page
        self.stacked_widget.setCurrentWidget(self.manual_input_page)

    def show_csv_upload(self):
        # Switch to the CSV upload page
        self.stacked_widget.setCurrentWidget(self.csv_upload_page)


def main():
    # Initialize the application
    app = QApplication(sys.argv)

    # Apply a global stylesheet for consistent look
    app.setStyleSheet("""
        QWidget {
            font-family: Arial, sans-serif;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
