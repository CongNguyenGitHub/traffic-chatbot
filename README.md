# 🚗 Chatbot Luật Giao Thông Việt Nam

## 📋 Tổng quan
Chatbot AI thông minh chuyên tư vấn về luật giao thông Việt Nam, sử dụng công nghệ RAG (Retrieval-Augmented Generation) để cung cấp thông tin chính xác từ Nghị định 168/2024/NĐ-CP.

## ✨ Tính năng chính
- 🤖 **Trả lời thông minh**: Sử dụng LLM tiếng Việt (Vistral-7B-Chat)
- 🔍 **Tìm kiếm ngữ nghĩa**: Vector embedding với Vietnamese-bi-encoder
- 📚 **Xử lý văn bản pháp lý**: Tách văn bản theo cấu trúc Mục-Điều-Khoản-Điểm
- 🚀 **Tối ưu hiệu suất**: 4-bit quantization, multiprocessing
- 🎨 **Giao diện thân thiện**: Streamlit web app
- 🛡️ **Xử lý lỗi**: Comprehensive error handling và logging

## 🏗️ Kiến trúc hệ thống

```
├── Documents (PDF) → Document Loader → Text Splitter
├── Text Chunks → Embeddings → Vector Store (FAISS)
├── User Query → Retriever → Context
└── Context + Query → LLM → Response
```
## 📸 Hình ảnh minh họa
<img src="assets\demo.png" alt="Traffic Law Chatbot Architecture" width="800"/>

## 🔧 Công nghệ sử dụng
- **LLM**: Viet-Mistral/Vistral-7B-Chat
- **Embeddings**: bkai-foundation-models/vietnamese-bi-encoder  
- **Framework**: LangChain, Transformers
- **Vector DB**: FAISS
- **UI**: Streamlit
- **Optimization**: BitsAndBytes (4-bit quantization)

## 🚀 Cài đặt và chạy

### 1. Clone repository
```bash
git clone <your-repo>
cd traffic-law-chatbot
```

###