# Use the official Python image
FROM python:3.12.8

# Set the working directory
WORKDIR /PHONEPREDICTIONSYSTEM

# Copy the application files
COPY ./Application /PHONEPREDICTIONSYSTEM/Application

# Copy the Random Forest PCA model
COPY ./Model/random_forest_pca_full_pipeline.joblib /PHONEPREDICTIONSYSTEM/Model/rf_pca_model.joblib

# Copy and install dependencies
COPY requirements.txt /PHONEPREDICTIONSYSTEM/requirements.txt
RUN pip install --no-cache-dir -r /PHONEPREDICTIONSYSTEM/requirements.txt

# Expose the FastAPI application's port
EXPOSE 80

# Run the FastAPI application
CMD ["uvicorn", "Application.back_end.fas_app:fas_app", "--host", "0.0.0.0", "--port", "80", "--reload"]