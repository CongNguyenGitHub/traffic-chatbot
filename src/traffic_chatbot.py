import os
import re
import logging
from typing import List, Dict, Optional
import multiprocessing
import glob
from tqdm import tqdm
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the chatbot"""
    EMBEDDING_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"
    LLM_MODEL: str = "Viet-Mistral/Vistral-7B-Chat"
    MAX_NEW_TOKENS: int = 128
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5

class TrafficLawTextSplitter:
    """Enhanced text splitter for Vietnamese legal documents"""
    
    def __init__(self):
        self.patterns = {
            'muc': re.compile(r"^Mục\s+\d+\..*", re.IGNORECASE),
            'dieu': re.compile(r"^Điều\s+\d+\..*", re.IGNORECASE),
            'khoan': re.compile(r"^\d+\.\s+.*"),
            'diem': re.compile(r"^[a-z]\)\s+.*")
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into legal structure chunks"""
        chunks = []
        
        for doc in documents:
            try:
                doc_chunks = self._process_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
                
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _process_document(self, doc: Document) -> List[Document]:
        """Process a single document"""
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        chunks = []
        
        current_context = {
            'muc': '',
            'dieu': '',
            'khoan': '',
            'diem': ''
        }
        
        i = 0
        while i < len(lines):
            line = lines[i]
            i = self._process_line(lines, i, current_context, chunks)
            
        return chunks
    
    def _process_line(self, lines: List[str], index: int, context: Dict[str, str], chunks: List[Document]) -> int:
        """Process a single line and update context"""
        line = lines[index]
        
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(line):
                return self._handle_pattern_match(lines, index, pattern_name, context, chunks)
        
        return index + 1
    
    def _handle_pattern_match(self, lines: List[str], index: int, pattern_type: str, 
                             context: Dict[str, str], chunks: List[Document]) -> int:
        """Handle pattern match and create chunks"""
        # Reset lower level contexts
        reset_levels = {
            'muc': ['dieu', 'khoan', 'diem'],
            'dieu': ['khoan', 'diem'],
            'khoan': ['diem'],
            'diem': []
        }
        
        for level in reset_levels[pattern_type]:
            context[level] = ''
        
        # Collect content for this pattern
        content_lines, next_index = self._collect_content(lines, index)
        context[pattern_type] = ' '.join(content_lines)
        
        # Create chunk if this is a complete unit (diem level)
        if pattern_type == 'diem':
            self._create_chunk(context, chunks)
        
        return next_index
    
    def _collect_content(self, lines: List[str], start_index: int) -> tuple:
        """Collect content lines for a pattern"""
        content_lines = [lines[start_index]]
        index = start_index + 1
        
        while index < len(lines):
            next_line = lines[index]
            if any(pattern.match(next_line) for pattern in self.patterns.values()):
                break
            content_lines.append(next_line)
            index += 1
            
        return content_lines, index
    
    def _create_chunk(self, context: Dict[str, str], chunks: List[Document]):
        """Create a document chunk from context"""
        full_text = "\n".join(filter(None, context.values()))
        if full_text.strip():
            chunks.append(Document(page_content=full_text.strip()))

class DocumentLoader:
    """Enhanced document loader with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.splitter = TrafficLawTextSplitter()
    
    def load_pdf_file(self, file_path: str) -> List[Document]:
        """Load a single PDF file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path, extract_images=False)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all PDF files from directory"""
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {directory_path}")
                return []
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            all_chunks = []
            
            for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
                chunks = self.load_pdf_file(pdf_file)
                all_chunks.extend(chunks)
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {e}")
            return []

class VectorStore:
    """Vector store manager with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            logger.info(f"Initializing embeddings: {self.config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create vector store from documents"""
        try:
            if not documents:
                logger.error("No documents provided for vector store creation")
                return False
            
            if self.embeddings is None:
                self._initialize_embeddings()
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def get_retriever(self):
        """Get retriever for RAG"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.TOP_K_RETRIEVAL}
        )

class LLMManager:
    """LLM manager with optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
    
    def initialize_llm(self, use_quantization: bool = True) -> bool:
        """Initialize the language model"""
        try:
            logger.info(f"Initializing LLM: {self.config.LLM_MODEL}")
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "use_cache": True,
            }
            
            if use_quantization and torch.cuda.is_available():
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = nf4_config
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                **model_kwargs
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline
            gen_pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                temperature=0.1,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            self.llm = HuggingFacePipeline(pipeline=gen_pipeline)
            logger.info("LLM initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False

class CustomOutputParser(StrOutputParser):
    """Custom output parser for Vietnamese responses"""
    
    def parse(self, text: str) -> str:
        """Parse and clean the model output"""
        try:
            # Extract answer after "Câu trả lời:"
            pattern = r"Câu trả lời:\s*(.*)$"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                answer = match.group(1).strip()
            else:
                # Fallback: return the text as is
                answer = text.strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            return answer
            
        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            return text.strip()
    
    def _clean_answer(self, text: str) -> str:
        """Clean the answer text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any remaining template artifacts
        text = re.sub(r'<[^>]+>', '', text)
        
        return text.strip()

class TrafficLawChatbot:
    """Main chatbot class"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.loader = DocumentLoader(self.config)
        self.vector_store = VectorStore(self.config)
        self.llm_manager = LLMManager(self.config)
        self.chain = None
        
        # Vietnamese prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Bạn là trợ lý AI chuyên về luật giao thông Việt Nam. Hãy trả lời câu hỏi một cách chính xác và hữu ích dựa trên thông tin được cung cấp.

Thông tin tham khảo từ Nghị định 168/2024/NĐ-CP:
{context}

Câu hỏi: {question}

Câu trả lời:"""
        )
    
    def initialize(self, documents_path: str) -> bool:
        """Initialize the chatbot"""
        try:
            logger.info("Initializing Traffic Law Chatbot...")
            
            # Load documents
            documents = self.loader.load_directory(documents_path)
            if not documents:
                logger.error("No documents loaded")
                return False
            
            # Create vector store
            if not self.vector_store.create_vector_store(documents):
                return False
            
            # Initialize LLM
            if not self.llm_manager.initialize_llm():
                return False
            
            # Create RAG chain
            self._create_rag_chain()
            
            logger.info("Chatbot initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            return False
    
    def _create_rag_chain(self):
        """Create the RAG chain"""
        retriever = self.vector_store.get_retriever()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm_manager.llm
            | CustomOutputParser()
        )
    
    def ask(self, question: str) -> str:
        """Ask a question to the chatbot"""
        try:
            if self.chain is None:
                return "Chatbot chưa được khởi tạo. Vui lòng gọi initialize() trước."
            
            logger.info(f"Processing question: {question}")
            response = self.chain.invoke(question)
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

