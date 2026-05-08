import gradio as gr
import joblib

# Load the trained model that we uploaded
model = joblib.load('spam_classifier_model.joblib')

# Define the function that the website will run
def classify_email(email_text):
    # Predict the class of the email
    prediction = model.predict([email_text])[0]
    
    # Format the output as requested
    if prediction.strip().lower() == 'ham':
        return "Safe Email (Ham) ✅"
    else:
        return "Spam 🚨"

# Create the visual layout using Gradio
iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=7, placeholder="Copy and paste an email here to test it..."),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Email Spam Classifier",
    description="This Machine Learning model was trained to differentiate between Spam and Safe Emails. Give it a try!",
    theme="default"
)

# Start the website
iface.launch()
