from dataclasses import dataclass, asdict
import datetime, time
import json
import os
import pickle
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np
import faiss

from raven.memory.memory import Memory, MemoryItem
from langchain_huggingface import HuggingFaceEmbeddings

FIXED_SUBTRACT = 1721761000  # this is just a large value that brings us close to 1970


class FAISSWrapper:
    """FAISS-based vector storage wrapper that replaces Milvus functionality"""
    
    def __init__(self, collection_name='test', storage_path='./faiss_storage', dim=1024, drop_collection=False, cache=False, l2_metric=False):
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.dim = dim
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # File paths for different indices and metadata
        self.text_index_path = os.path.join(storage_path, f"{collection_name}_text.index")
        self.position_index_path = os.path.join(storage_path, f"{collection_name}_position.index")
        self.time_index_path = os.path.join(storage_path, f"{collection_name}_time.index")
        self.metadata_path = os.path.join(storage_path, f"{collection_name}_metadata.json")
        self.id_counter_path = os.path.join(storage_path, f"{collection_name}_id_counter.json")
        
        # Initialize indices
        self.text_index = None
        self.position_index = None
        self.time_index = None
        self.metadata = []
        self.id_counter = 0
        
        # Load existing data or create new
        if drop_collection:
            self.drop_collection()

        self.cache = cache
        self.l2_metric = l2_metric
        self._load_or_create_indices(cache)
    
    def drop_collection(self):
        """Remove all files for this collection"""
        for path in [self.text_index_path, self.position_index_path, self.time_index_path, 
                    self.metadata_path, self.id_counter_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def _load_or_create_indices(self, load=False):
        """Load existing indices or create new ones"""
        # Load metadata
        if load and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
        
        # Load ID counter
        if load and os.path.exists(self.id_counter_path):
            with open(self.id_counter_path, 'r') as f:
                self.id_counter = json.load(f)
        else:
            self.id_counter = 0
        
        # Create or load FAISS indices
        self._create_or_load_index('text', self.text_index_path, self.dim, load)
        self._create_or_load_index('position', self.position_index_path, 3, load)
        self._create_or_load_index('time', self.time_index_path, 2, load)
    
    def _create_or_load_index(self, index_type, path, dim, load=False):
        """Create new FAISS index or load existing one"""
        if load and os.path.exists(path):
            # Load existing index
            if index_type == 'text':
                self.text_index = faiss.read_index(path)
            elif index_type == 'position':
                self.position_index = faiss.read_index(path)
            elif index_type == 'time':
                self.time_index = faiss.read_index(path)
        else:
            # Create new index
            if index_type == 'text':
                self.text_index = faiss.IndexFlatL2(dim) if self.l2_metric else faiss.IndexFlatIP(dim) 
            elif index_type == 'position':
                self.position_index = faiss.IndexFlatL2(dim)
            elif index_type == 'time':
                self.time_index = faiss.IndexFlatL2(dim)
    
    def _save_indices(self):
        """Save all indices to disk"""
        if self.text_index is not None:
            faiss.write_index(self.text_index, self.text_index_path)
        if self.position_index is not None:
            faiss.write_index(self.position_index, self.position_index_path)
        if self.time_index is not None:
            faiss.write_index(self.time_index, self.time_index_path)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        # Save ID counter
        with open(self.id_counter_path, 'w') as f:
            json.dump(self.id_counter, f)
    
    def insert(self, data_list):
        """Insert data into all indices"""
        for data in data_list:
            # Generate unique ID
            data['id'] = str(self.id_counter)
            self.id_counter += 1
            
            # Add to metadata
            self.metadata.append(data)
            
            # Add vectors to indices
            if 'text_embedding' in data:
                text_vector = np.array([data['text_embedding']], dtype=np.float32)
                self.text_index.add(text_vector)
            
            if 'position' in data:
                position_vector = np.array([data['position']], dtype=np.float32)
                self.position_index.add(position_vector)
            
            if 'time' in data:
                time_vector = np.array([data['time']], dtype=np.float32)
                self.time_index.add(time_vector)
        
        # Save to disk
        if self.cache:
            self._save_indices()
    
    def search_by_text_embedding(self, query_vector, k):
        """Search by text embedding vector"""
        if self.text_index.ntotal == 0:
            return []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.text_index.search(query_vector, min(k, self.text_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                # Create Document object similar to Milvus results
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results
    
    def search_by_position(self, position_vector, k):
        """Search by position vector"""
        if self.position_index.ntotal == 0:
            return []
        
        query_vector = np.array([position_vector], dtype=np.float32)
        distances, indices = self.position_index.search(query_vector, min(k, self.position_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results
    
    def search_by_time(self, time_vector, k):
        """Search by time vector"""
        if self.time_index.ntotal == 0:
            return []
        
        query_vector = np.array([time_vector], dtype=np.float32)
        distances, indices = self.time_index.search(query_vector, min(k, self.time_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results


class FAISSMemory(Memory):
    """FAISS-based memory implementation that replaces MilvusMemory"""
    
    def __init__(self, db_collection_name: str, embedder: HuggingFaceEmbeddings, storage_path='./faiss_storage', time_offset=FIXED_SUBTRACT, 
                 use_vlm_embedding=False, vlm_model_path=None, dim=1024, retriever_k=5):
        
        self.db_collection_name = db_collection_name
        self.storage_path = storage_path
        self.time_offset = time_offset
        self.use_vlm_embedding = use_vlm_embedding
        self.dim = dim
        self.retriever_k = retriever_k

        # Initialize embedder
        self.embedder = embedder
        print("Using HuggingFace text embedding with FAISS storage") 

        self.working_memory = []
        self.reset(drop_collection=False)
        assert isinstance(self.retriever_k, (int, dict)), "retriever_k must be an int or a dict of ints."

    def insert(self, item: MemoryItem, text_embedding=None, image=None):
        """Insert a memory item"""
        memory_dict = asdict(item)
        memory_dict['id'] = str(time.time())

        if text_embedding is None:
            if self.use_vlm_embedding and hasattr(item, 'image') and item.image is not None:
                # Use VLM to generate embedding from image
                text_embedding = self.embedder.embed_image(item.image)
            elif self.use_vlm_embedding and image is not None:
                # Use provided image for VLM embedding
                text_embedding = self.embedder.embed_image(image)
            else:
                # Fallback to text embedding
                text_embedding = self.embedder.embed_query("[TXT]"+memory_dict['caption'])

        memory_dict['time'] = [(memory_dict['time'] - self.time_offset), 0.0]
        memory_dict['text_embedding'] = text_embedding

        self.faiss_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):
        """Reset the memory storage"""
        if drop_collection:
            print("Resetting FAISS memory. Dropping current collection")

        self.faiss_wrapper = FAISSWrapper(
            collection_name=self.db_collection_name,
            storage_path=self.storage_path,
            dim=self.dim,
            drop_collection=drop_collection
        )

    def search_by_position(self, query: tuple) -> str:
        """Search memories by position"""
        docs_with_scores = self.faiss_wrapper.search_by_position(np.array(query).astype(float), self.retriever_k["position"] if isinstance(self.retriever_k, dict) else self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string(docs)
        return docs

    def search_by_time(self, hms_time: str) -> str:
        """Search memories by time"""
        # Convert time string to searchable format
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # Check if time is already in correct format
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res:
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple()) - self.time_offset
        query_vector = np.array([query, 0])

        docs_with_scores = self.faiss_wrapper.search_by_time(query_vector, self.retriever_k["time"] if isinstance(self.retriever_k, dict) else self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string(docs)
        return docs

    def search_by_text(self, query: str) -> str:
        """Search memories by text"""
        # Generate embedding for the query
        query_embedding = self.embedder.embed_query("[TXT]" + query)
        
        docs_with_scores = self.faiss_wrapper.search_by_text_embedding(query_embedding, self.retriever_k["text"] if isinstance(self.retriever_k, dict) else self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string(docs)
        return docs

    def memory_to_string(self, memory_list: list[Document], ref_time: float = None):
        """Convert memory documents to string format"""
        if ref_time is None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string
