from flask import Flask, request, jsonify, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex
import time
import textwrap

app = Flask(__name__)

# Load data and initialize models
hn_filepath = 'Gita.xlsx'
hn_data = pd.read_excel(hn_filepath)

hn_model = SentenceTransformer('all-mpnet-base-v2')
shlok_keys = ['Title', 'Chapter', 'Verse','Sanskrit Anuvad', 'Hindi Anuvad', 'Enlgish Translation']

shloka_embeddings = [hn_model.encode(hn_data['Enlgish Translation'][i], convert_to_tensor=False) for i in range(len(hn_data))]

embedding_size = len(shloka_embeddings[0])
annoy_index = AnnoyIndex(embedding_size, 'angular')

# for i, embedding in enumerate(shloka_embeddings):
#   annoy_index.add_item(i, embedding)
#   if i % 20 == 0:
#     print(f"Building Annoy Index: {i}/{len(shloka_embeddings)}")

# annoy_index.build(18) 
# print("Annoy index built successfully.")

# Save the index to a file
# annoy_index.save('annoy_index.ann')
# print("Annoy index saved successfully.")

annoy_index.load('annoy_index.ann')
print("Annoy index loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_embedding = hn_model.encode(query, convert_to_tensor=True).numpy()

    # Use Annoy Index for efficient similarity search
    similar_indices = annoy_index.get_nns_by_vector(query_embedding, 18)

    # Process and Display Similar Shlokas
    similarities = []
    for curr_index in similar_indices:
        similarity = util.cos_sim(query_embedding, shloka_embeddings[curr_index])
        curr_shlok_details = {key: hn_data[key][curr_index] for key in shlok_keys}
        similarities.append({"shlok_details": curr_shlok_details, "similarity": similarity})

    # Get the most similar Shlok
    top_result = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[0]
    top_shlok_details = top_result["shlok_details"]
    adhyay_number = top_shlok_details["Chapter"].split(" ")[1]
    shlok_number = top_shlok_details['Verse'].split(" ")[1].split(".")[1]

    meaning_text = top_shlok_details['Enlgish Translation']
    wrapped_text = textwrap.fill(meaning_text, width=80)
    wrapped_hindi_text = textwrap.fill(top_shlok_details['Hindi Anuvad'], width=80)
    wrapped_sanskrit_text = textwrap.fill(top_shlok_details['Sanskrit Anuvad'], width=80)

    response = {
        "chapter": top_shlok_details['Chapter'],
        "verse": shlok_number,
        "english_translation": wrapped_text,
        "hindi_translation": wrapped_hindi_text,
        "sanskrit_translation": wrapped_sanskrit_text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


