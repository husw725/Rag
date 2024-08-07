from flask import Flask, request
from local_doc_qa import *
import os

app = Flask(__name__)

local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg()

vs_id = "agent"
vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
filelist = []
filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", f"{vs_id}.txt"))
print(filelist[0])

if not os.path.exists(os.path.join(vs_path, "index.faiss")):
    local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size=100)    

@app.route('/question', methods=['GET'])
def ask_question():
    question = request.args.get('q')  # Assuming the question is sent in the query parameters

    docs = local_doc_qa.get_knowledge_based_answer(query=question, vs_path=vs_path)
    response = "\n".join(doc.page_content for doc in docs)

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9094)