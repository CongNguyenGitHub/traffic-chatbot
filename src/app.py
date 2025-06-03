import streamlit as st
import os
from traffic_chatbot import TrafficLawChatbot, Config
from huggingface_hub import login 
from dotenv import load_dotenv

# Load variables from env file 
load_dotenv()

# Login huggingface using tokens
login(os.getenv("API_KEY"))

# Configure page
st.set_page_config(
    page_title="Chatbot Luật Giao Thông Việt Nam",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #007bff;
    color: white;
    padding: 0.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: right;
}
.bot-message {
    background-color: #28a745;
    color: white;
    padding: 0.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Header
st.markdown('<h1 class="main-header">🚗 Chatbot Luật Giao Thông Việt Nam</h1>', unsafe_allow_html=True)
st.markdown("*Dựa trên Nghị định 168/2024/NĐ-CP về xử phạt hành chính trong lĩnh vực giao thông đường bộ*")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Data folder configuration
    st.subheader("📁 Cấu hình dữ liệu")
    data_folder = st.text_input(
        "Đường dẫn folder chứa dữ liệu:",
        value="./data",
        help="Nhập đường dẫn tới folder chứa các file PDF luật giao thông"
    )
    
    # Check if folder exists and show files
    if os.path.exists(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        if pdf_files:
            st.success(f"📄 Tìm thấy {len(pdf_files)} file PDF:")
            for file in pdf_files:
                st.write(f"• {file}")
        else:
            st.warning("⚠️ Không tìm thấy file PDF nào trong folder")
    else:
        st.error("❌ Folder không tồn tại")
    
    # Model configuration
    st.subheader("🤖 Cấu hình Model")
    max_tokens = st.slider("Max tokens", 100, 512, 256)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.1)
    top_k = st.slider("Top K retrieval", 3, 10, 5)
    
    # Initialize button
    if st.button("🚀 Khởi tạo Chatbot", type="primary"):
        if os.path.exists(data_folder):
            pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
            if pdf_files:
                with st.spinner("Đang khởi tạo chatbot..."):
                    try:
                        # Initialize chatbot with config
                        config = Config()
                        config.MAX_NEW_TOKENS = max_tokens
                        config.TOP_K_RETRIEVAL = top_k
                        
                        chatbot = TrafficLawChatbot(config)
                        
                        if chatbot.initialize(data_folder):
                            st.session_state.chatbot = chatbot
                            st.session_state.initialized = True
                            st.success("✅ Chatbot đã được khởi tạo thành công!")
                        else:
                            st.error("❌ Lỗi khởi tạo chatbot")
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
            else:
                st.warning("⚠️ Không tìm thấy file PDF nào trong folder")
        else:
            st.error("❌ Folder không tồn tại. Vui lòng kiểm tra đường dẫn.")

# Main chat interface
if st.session_state.initialized:
    st.success("🤖 Chatbot đã sẵn sàng! Hãy đặt câu hỏi về luật giao thông.")
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("💬 Lịch sử trò chuyện")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f'<div class="user-message">👤 {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">🤖 {answer}</div>', unsafe_allow_html=True)
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Đặt câu hỏi của bạn:",
            placeholder="VD: Mức phạt khi không đội mũ bảo hiểm là bao nhiêu?",
            help="Hãy đặt câu hỏi cụ thể về luật giao thông"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.form_submit_button("Gửi", type="primary")
        with col2:
            clear_button = st.form_submit_button("Xóa chat")
    
    # Handle form submission
    if submit_button and user_input:
        with st.spinner("Đang tìm kiếm thông tin..."):
            try:
                response = st.session_state.chatbot.ask(user_input)
                st.session_state.chat_history.append((user_input, response))
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

else:
    # Instructions
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("📋 **Hướng dẫn sử dụng:**")
    st.markdown("""
    1. **Chuẩn bị dữ liệu**: Đặt các file PDF chứa quy định giao thông vào folder local
    2. **Cấu hình folder**: Nhập đường dẫn tới folder chứa dữ liệu trong sidebar
    3. **Kiểm tra**: Hệ thống sẽ tự động kiểm tra và hiển thị các file PDF tìm thấy
    4. **Cấu hình model**: Điều chỉnh các thông số model nếu cần
    5. **Khởi tạo**: Click nút "Khởi tạo Chatbot"
    6. **Trò chuyện**: Đặt câu hỏi về luật giao thông Việt Nam
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Folder structure example
    st.subheader("📁 Cấu trúc thư mục đề xuất")
    st.code("""
    project/
    ├── src/
        ├── app.py
        └── traffic_chatbot.py
    └── data/
        ├── nghi_dinh_168_2024.pdf
        ├── luat_giao_thong_duong_bo.pdf
        └── quy_dinh_khac.pdf
    
    """, language="text")
    
    # Sample questions
    st.subheader("🔍 Câu hỏi mẫu")
    sample_questions = [
        "Mức phạt khi vượt đèn đỏ là bao nhiêu?",
        "Phạt bao nhiều tiền khi không đội mũ bảo hiểm?",
        "Các trường hợp bị tước bằng lái xe?",
        "Quy định về nồng độ cồn khi lái xe?",
        "Mức phạt khi sử dụng điện thoại khi lái xe?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        st.write(f"{i}. {question}")

# Footer
st.markdown("---")
st.markdown("*Được phát triển với ❤️ bằng Streamlit và LangChain*")
st.markdown("⚠️ *Lưu ý: Thông tin chỉ mang tính chất tham khảo. Vui lòng tham khảo văn bản pháp luật chính thức.*")