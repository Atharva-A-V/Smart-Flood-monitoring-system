# Smart-Flood-monitoring-system

create requirments.txt file having : 
numpy
streamlit
matplotlib
seaborn
pandas
Pillow
python-dotenv
rasterio
scikit-image
sentence-transformers
torch
open-clip-torch
kagglehub
openai
geopy
google-generativeai

create a .env file having the api key for the LLM (done to keep the api key safe )
create a virtual environment using : python -m venv venv
activate the virtual environment using : .\venv\Scripts\Activate.ps1
install packages from requirements.txt : pip install -r requirements.txt
after done, to launch project on streamlit simply use : streamlit run filename.py ( filename is name of the RAG file )
install a tif image from google earth engine
upload it in streamlit app to get the results
