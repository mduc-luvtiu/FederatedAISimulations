## **1. File `main.py`**

```python
from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate 
from vector import retriever

model = OllamaLLM(model="qwen3:0.6b") 
```

* Import các thư viện cần thiết:

  * `OllamaLLM`: để gọi LLM từ Ollama (ở đây là model `qwen3:0.6b`).
  * `ChatPromptTemplate`: để định nghĩa template prompt.
  * `retriever`: được định nghĩa ở file `vector.py` (retrieval từ database embedding).

```python
template = """
You are an expert in answering questions about a pizza restaurant 

Here are some relevant reviews: {reviews} 

Here is the question to answer: {questions}
"""
```

* Prompt template: mô hình sẽ đóng vai **chuyên gia trả lời về nhà hàng pizza**.
* Biến `{reviews}` sẽ chứa các review tìm được từ DB.
* Biến `{questions}` là câu hỏi người dùng nhập.

```python
prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model 
```

* `prompt` là một object của LangChain để format template trên.
* `chain = prompt | model`: dùng “pipe operator” (`|`) để tạo ra chain. Nghĩa là dữ liệu sẽ đi qua `prompt` (chèn text) rồi được gửi cho `model`.

```python
while True:
    print("-"*50)
    question = input("\n\nAsk your question (q to quit): ")
    if question == "q": 
        break
    
    reviews = retriever.invoke(question)
    
    result = chain.invoke({"reviews": reviews, "questions": question})
    
    print(result)
```

* Chạy vòng lặp liên tục để nhận câu hỏi người dùng.
* Nếu gõ `"q"` thì thoát.
* `retriever.invoke(question)`: gọi retriever để lấy ra các review liên quan đến câu hỏi.
* `chain.invoke(...)`: nhét reviews + câu hỏi vào template prompt, gửi cho LLM để sinh câu trả lời.
* In ra kết quả.

**Ý tưởng:**
File `main.py` là chương trình chat. Người dùng hỏi, hệ thống đi tìm các review liên quan trong DB embedding, rồi đưa vào prompt để LLM trả lời.

---

## **2. File `vector.py`**

```python
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.documents import Document 
import os 
import pandas as pd 
```

* Import công cụ embedding (`OllamaEmbeddings`), database vector (`Chroma`), class `Document` để lưu văn bản, cùng `os` và `pandas`.

```python
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
```

* Đọc dữ liệu từ file CSV chứa review nhà hàng.
* Khởi tạo model embedding (`mxbai-embed-large:335m`) để nhúng văn bản thành vector.

```python
db_location = "./chrome_langchain_db" 
add_documents = not os.path.exists(db_location) 
```

* Đặt thư mục lưu vector database.
* Nếu thư mục chưa tồn tại (`add_documents=True`), thì sẽ tạo và thêm dữ liệu mới.

```python
if add_documents: 
    documents = []
    ids = []  
    
    for i, row in df.iterrows(): 
        document = Document(
            page_content=row["Title"] + row["Review"], 
            metadata = {"rating": row["Rating"], "date": row["Date"]}, 
            id=str(i), 
        )
        ids.append(str(i)) 
        documents.append(document)
```

* Nếu DB chưa tồn tại:

  * Duyệt từng dòng dữ liệu trong CSV.
  * Tạo `Document` gồm:

    * `page_content`: gộp **Title + Review** để làm nội dung chính.
    * `metadata`: chứa thêm thông tin (rating, date).
    * `id`: số thứ tự.
  * Lưu vào list `documents` và `ids`.

```python
vector_store = Chroma(
    collection_name="restaurant_reviews", 
    persist_directory=db_location, 
    embedding_function=embeddings
)
```

* Tạo một vector store bằng **ChromaDB**.
* Lưu trong thư mục `chrome_langchain_db`.
* Dùng hàm nhúng `embeddings` để xử lý text.

```python
if add_documents: 
    vector_store.add_documents(documents=documents, ids=ids) 
```

* Nếu đây là lần đầu (DB chưa có), thì thêm toàn bộ review từ CSV vào vector store.

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
```

* Biến `retriever` chính là API để tìm **5 review gần nhất (theo vector similarity)** với câu hỏi của người dùng.
* Đây là thứ được import trong `main.py`.

---

## **Tóm tắt luồng hoạt động**

1. Người dùng nhập câu hỏi trong `main.py`.
2. `retriever.invoke(question)` → tìm 5 review gần nhất trong CSV (đã embed trong ChromaDB).
3. Gắn các review này + câu hỏi vào `prompt`.
4. Gửi cho model `qwen3:0.6b` qua `OllamaLLM`.
5. Model trả về câu trả lời mang tính chuyên gia (dựa trên dữ liệu review).

Đây là một **hệ thống Question Answering (QA) theo ngữ cảnh**, dùng LangChain + Ollama + ChromaDB, áp dụng cho dữ liệu review nhà hàng pizza.
