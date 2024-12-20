# Use the official Python image
FROM python:3.12.8

# Set the working directory
WORKDIR /PHONEPREDICTIONSYSTEM/Application/back_end

# Copy the necessary files
COPY ./Application /PHONEPREDICTIONSYSTEM/Application
COPY ./Model/xgboost_classifier_model_ultimate.pkl /PHONEPREDICTIONSYSTEM/Model/xgb_model.pkl

# Install dependencies
COPY requirements.txt /PHONEPREDICTIONSYSTEM/requirements.txt
RUN pip install -r /PHONEPREDICTIONSYSTEM/requirements.txt

# Run the application
CMD ["uvicorn", "Application.fas_app:fas_app", "--host", "0.0.0.0", "--port", "80", "--reload"]
