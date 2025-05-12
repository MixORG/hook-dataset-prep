<body>

1. git clone https://github.com/MixORG/hook-dataset-prep.git<br>
2. cd hook-dataset-prep<br>
3. python -m venv hook-dataset-prep<br>
4. source hook-dataset-prep/bin/activate<br>
5. pip install requirements.txt<br>
6. "Put your Deepgram API key in video_dataset_pipeline.py file at line number 11"<br>
7. Save file video_dataset_pipeline.py<br>
8. Fill in YouTube URLs in sample_urls.csv<br>
9. python video_dataset_pipeline.py sample_urls.csv \
    --work-dir downloads \
    --cookies-from-browser chrome
</body>
