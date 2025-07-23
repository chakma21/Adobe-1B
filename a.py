from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save_pretrained('C:/Users/tokos/Desktop/project/adobe/models/all-MiniLM-L6-v2')
print("Model downloaded and saved.")