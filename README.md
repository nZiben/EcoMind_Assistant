# Description of the solution for the environmental analysis project

### Main task
Creation of a module for automatic analysis of pollution sources and comparison with regulatory indicators. Advanced methods of specialized data processing were used for this purpose.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Solution
1. **Additional training for Yandex.GPT-PRO**  
   Creating a large dataset of questions and answers using Yandex.Maps for educational and regulatory materials.

2. **Creating a knowledge base**  
   Vectorization of standards and textbooks using Open-Source Encoder-only LLM and storage in the ChromaDB database.

3. **Table processing**  
   The source files are converted to Web Layout to combine the torn tables. The tables are extracted using python-docx, then the Yandex.gpt models summarize them and save them to .txt.

4. **Multi-layered RAG**  
   Two vector databases have been built for the knowledge base and the uploaded data. The RAG system with two retrievers and Yandex.GPT-PRO generates responses based on data and regulations.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Work results
The system automates the analysis of environmental reports, reducing the impact of the human factor and increasing the accuracy of data processing. This allows businesses and regulators to identify environmental risks faster and take timely action.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### File Description
- eco_QnA_dataset.jsonl - dataset generated using YandexGPT from the attached sources and used for further training

- documents/RAG - the knowledge base used to create the context for the query

- src/summarization.ipynb - laptop with summarization

- src/main.py - a script for using a pre-trained model in synchronous mode

- src/ecology_api.py - script for accessing the model via API
