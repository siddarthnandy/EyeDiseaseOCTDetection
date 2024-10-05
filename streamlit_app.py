import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Load the trained model
model = tf.keras.models.load_model('model150.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to fit the model input
    image_array = np.array(image).astype('float32') / 255.0  # Normalize to [0, 1]
    return image_array

# Streamlit app
st.title('OCT Eye Disease Detection Report')

# Input fields for report details
patient_name = st.text_input("Patient Name:")
patient_id = st.text_input("Patient ID:")
doctor_name = st.text_input("Doctor Name:")
report_date = st.date_input("Date of Report:")

# Upload images for the right and left eyes
uploaded_file_right = st.file_uploader("Upload Right Eye Image (OCT)...", type=["jpg", "jpeg", "png"], key='right_eye')
uploaded_file_left = st.file_uploader("Upload Left Eye Image (OCT)...", type=["jpg", "jpeg", "png"], key='left_eye')

# Define class names and severity levels
class_names = ['0', '1', '2', '3']  # CNV, DME, Drusen, Normal
severity_levels = {
    '0': 'Choroidal Neovascularization (CNV)',
    '1': 'Diabetic Macular Edema (DME)',
    '2': 'Drusen',
    '3': 'Normal'
}

def predict_eye(image):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class]

# Placeholders for predictions
prediction_right = None
prediction_left = None

# Display images
if uploaded_file_right is not None:
    st.subheader('Right Eye Image (OCT)')
    image_right = Image.open(uploaded_file_right)
    st.image(image_right, caption='Uploaded Right Eye Image', use_column_width=True)
    prediction_right = predict_eye(image_right)

if uploaded_file_left is not None:
    st.subheader('Left Eye Image (OCT)')
    image_left = Image.open(uploaded_file_left)
    st.image(image_left, caption='Uploaded Left Eye Image', use_column_width=True)
    prediction_left = predict_eye(image_left)

# Display predictions and generate report at the bottom
if prediction_right is not None or prediction_left is not None:
    st.subheader('Predictions')
    if prediction_right is not None:
        st.write(f"Prediction for Right Eye: {prediction_right} - {severity_levels[prediction_right]}")
    if prediction_left is not None:
        st.write(f"Prediction for Left Eye: {prediction_left} - {severity_levels[prediction_left]}")

    # Generate the report as a PDF
    def generate_pdf_report(image_right, image_left, prediction_right, prediction_left, patient_name, patient_id, doctor_name, report_date):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, height - 50, "OCT Eye Disease Detection Report")

        # Report details
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Name: {patient_name}")
        c.drawString(300, height - 80, f"Date: {report_date}")
        c.drawString(50, height - 100, f"Patient ID: {patient_id}")
        c.drawString(300, height - 100, f"Doctor: {doctor_name}")

        # Images and labels
        c.drawString(100, height - 150, "Right Eye Image (OCT)")
        c.drawString(350, height - 150, "Left Eye Image (OCT)")
        c.drawImage(ImageReader(image_right), 50, height - 400, width=3*inch, height=3*inch)
        c.drawImage(ImageReader(image_left), 300, height - 400, width=3*inch, height=3*inch)

        # Results
        c.drawString(50, height - 450, "Results")
        c.line(50, height - 460, width - 50, height - 460)
        c.drawString(50, height - 480, "Eye Disease Severity:")
        c.drawString(250, height - 480, f"Right Eye: {severity_levels[prediction_right]}")
        c.drawString(450, height - 480, f"Left Eye: {severity_levels[prediction_left]}")

        # Disclaimers
        c.drawString(50, height - 540, "Disclaimers:")
        c.drawString(50, height - 560, "This report does not replace professional medical advice, diagnosis, or treatment.")
        c.drawString(50, height - 580, "Clinician notes:")

        c.save()
        buffer.seek(0)
        return buffer

    # Provide a download button
    if st.button("Download OCT Eye Disease Detection Report"):
        pdf_report = generate_pdf_report(image_right, image_left, prediction_right, prediction_left, patient_name, patient_id, doctor_name, report_date)
        st.download_button(label="Download Report", data=pdf_report, file_name="OCT_Eye_Disease_Detection_Report.pdf", mime="application/pdf")
