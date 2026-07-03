import base64, mimetypes, pathlib
import os, pathlib
import torch
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from PIL import Image
import pickle as pkl
from transformers.image_utils import load_image
from qwen_vl_utils import process_vision_info
from langchain_core.embeddings import Embeddings
import numpy as np
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoProcessor
)
from transformers import pipeline
from volcenginesdkarkruntime import Ark
import vertexai
from vertexai.vision_models import Image as VImage, MultiModalEmbeddingModel

MODEL_NAME_MAPPING = {
    "seed1.6": "doubao-embedding-vision-250615",
    "seed": "doubao-embedding-vision-250615",
    "google": "multimodalembedding@001",
}


def safe_image_open(path: str):
    img = Image.open(path).convert("RGB")
    return img


def file_to_data_url(path: str) -> str:
    """convert file to data URL, support pkl and image file"""
    p = pathlib.Path(path)
    
    try:
        if p.suffix.lower() == '.pkl':
            # process pkl file
            import pickle
            from PIL import Image
            import io
            
            with open(p, 'rb') as f:
                data = pickle.load(f)
            
            # check pkl file structure
            if not isinstance(data, dict):
                raise ValueError(f"PKL file {path} should contain a dictionary")
            
            if 'cam0' not in data:
                raise ValueError(f"PKL file {path} does not contain 'cam0' key")
            
            img_array = data['cam0']
            
            # validate image data
            if not isinstance(img_array, np.ndarray):
                raise ValueError(f"'cam0' should be a numpy array, got {type(img_array)}")
            
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Expected image shape (H, W, 3), got {img_array.shape}")
            
            # ensure data type is correct
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            # convert to PIL image
            img = Image.fromarray(img_array, 'RGB')
            # convert to base64
            buffer = io.BytesIO()
            # Change the quality if seed cannot handle the image size
            img.save(buffer, format='JPEG', quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            
            return f"data:image/jpeg;base64,{b64}"
        
        else:
            # process normal image file
            mime, _ = mimetypes.guess_type(p.name)
            mime = mime or "image/jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
            
    except Exception as e:
        raise ValueError(f"Failed to convert {path} to data URL: {e}")

def load_image_from_pkl_or_path(path):
    """load image from pkl file or path"""
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        img_array = data['cam0']  # shape: (1024, 1224, 3)
        return Image.fromarray(img_array.astype('uint8'), 'RGB')
    else:
        return safe_image_open(path)

class VLMEmbeddings(Embeddings):
    def __init__(self,
                 device, 
                 backend, 
                 oc_model, 
                 oc_pretrained, 
                 hf_model_id, 
                 vlm_layer, 
                 batch_size,
                 fp_16: bool,
                 emb_dim: int, 
                 vlm_text_prompts, 
                 vlm_image_prompts,
                 online_model_nickname):
        super().__init__()
        # Better CUDA availability check
        if device == "cuda":
            try:
                if not torch.cuda.is_available():
                    print("⚠️  CUDA not available, falling back to CPU")
                    device = "cpu"
                else:
                    # Try to actually use CUDA to catch initialization errors
                    _ = torch.tensor([1.0]).cuda()
                    print(f"✓ CUDA is available and working. Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed: {e}")
                print("⚠️  Falling back to CPU")
                device = "cpu"
        
        # Create model / transforms. For SigLIP variants this will pull the right HF tokenizer internally if needed.
        if backend == "oc":
            model, _, preprocess = open_clip.create_model_and_transforms(
                oc_model, pretrained=oc_pretrained, device=device, cache_dir=".cache/openclip"
            )
            model.visual.preprocess = preprocess  # for convenience in our helpers
            text_tokenizer = open_clip.get_tokenizer(oc_model, cache_dir=".cache/openclip")
            model.eval()
        
        elif backend == "hf":
            if "paligemma" in hf_model_id:
                print("using layer=", vlm_layer)
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                                                            hf_model_id, 
                                                            torch_dtype=torch.bfloat16,
                                                            device_map=device,
                                                        ).eval()
                preprocess = PaliGemmaProcessor.from_pretrained(hf_model_id, use_fast=True)
                model.eval()
                
            elif "Qwen" in hf_model_id:
                print("using layer=", vlm_layer)
                from transformers import Qwen2_5_VLForConditionalGeneration
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                                            hf_model_id, 
                                                            torch_dtype="auto", 
                                                            device_map="auto"
                                                        ).eval()
                preprocess = AutoProcessor.from_pretrained(hf_model_id, use_fast=True)
                model.eval()

            elif "dinov3" in hf_model_id:
                model = pipeline(
                    model=hf_model_id,
                    task="image-feature-extraction",
                    device_map="auto"
                )
                preprocess = None

            elif "QQMM" in hf_model_id:
                from qqmm.models import build_processor
                from qqmm.utils.parameter_manage import Parameters
                from qqmm.models.qqmm_nav_qwen2.modeling_qqmm import QQMMForCausalLM
                from qqmm.utils.chat import EmbedBot
                config_qqmm = Parameters()
                # HEADSUP: CONFIG ADAPTATION FOR LOCAL IMPLEMENTATION
                config_path = Path(__file__).parent.parent.parent / "external/qqmm-package/configs/embed/qqmm-embed/mmeb.yaml"
                config_qqmm.merge_from_yaml(str(config_path))
                config_qqmm.PROCESSOR_CONFIG["tokenizer_path"] = str(Path(__file__).parent.parent.parent / "external/qqmm-package" / config_qqmm.PROCESSOR_CONFIG["tokenizer_path"])
                config_qqmm.PROCESSOR_CONFIG["image_processor_path"] = str(Path(__file__).parent.parent.parent / "external/qqmm-package" / config_qqmm.PROCESSOR_CONFIG["image_processor_path"])

                print(">>> Building Model...")
                processor = build_processor(config_qqmm.PROCESSOR_CONFIG, inferring=True)
                try:
                    print(">>> Loading Model Locally...", hf_model_id)
                    model_qqmm = QQMMForCausalLM.from_pretrained(
                        hf_model_id, 
                        torch_dtype=torch.bfloat16, 
                        device_map=device, 
                        local_files_only=True
                    )
                except Exception as e:
                    print(f"Local loading failed ({e}), downloading from HuggingFace...")
                    model_qqmm = QQMMForCausalLM.from_pretrained(
                        hf_model_id, 
                        torch_dtype=torch.bfloat16, 
                        device_map=device
                    )
                print(">>> Model Ready!")
                model = EmbedBot(model_qqmm, processor)
                preprocess = None
                print(">>> ChatBot Ready!")

            else:
                raise NotImplementedError

            text_tokenizer = None
        
        elif backend == "ol":
            text_tokenizer = None
            if online_model_nickname is None:
                raise ValueError("online_model_nickname must be provided for 'ol' backend.")
            
            if "seed" in online_model_nickname:
                seed_api_key = os.environ["SEED_API_KEY"] 
                model = Ark(api_key=seed_api_key)
                preprocess = None
            elif "google" in online_model_nickname:
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    raise RuntimeError(
                        "GOOGLE_CLOUD_PROJECT env var is required for the Google Vertex "
                        "embedder. Set it to your own GCP project ID (e.g. "
                        "`export GOOGLE_CLOUD_PROJECT=my-project`)."
                    )
                location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                vertexai.init(project=project_id, location=location)
                model = MultiModalEmbeddingModel.from_pretrained(MODEL_NAME_MAPPING[online_model_nickname])
                preprocess = None
            else:
                raise NotImplementedError
                    
        else:
            raise NotImplementedError

        if fp_16 and device == "cuda":
            # Many SigLIP models work fine in fp16; if you see instabilities, run in fp32
            model = model.half()

        self.model = model
        self.preprocess = preprocess
        self.text_tokenizer = text_tokenizer
        
        self.backend = backend
        self.device = device
        self.oc_model=oc_model
        self.oc_pretrained = oc_pretrained
        self.hf_model_id = hf_model_id
        self.vlm_layer = vlm_layer
        self.fp_16 = fp_16
        self.vlm_image_prompts = vlm_image_prompts
        self.vlm_text_prompts = vlm_text_prompts
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.online_model_nickname = online_model_nickname


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.
        Args:
            texts: List of image filenames, or texts to embed.
        Returns:
            List of embeddings.
        """
        text_num = len([0 for text in texts if text[:5] == "[TXT]"])
        id_t, id_i = 0, text_num
        idxs_ = []
        texts_ = []
        images_ = []
        for i, text in enumerate(texts):
            if text[:5] == "[TXT]":
                idxs_.append(id_t)
                texts_.append(text[5:])
                id_t += 1
            elif text[:5] == "[IMG]":
                idxs_.append(id_i)
                images_.append(text[5:])
                id_i += 1
            else: raise ValueError("input text should be starting with either [TXT] or [IMG]!")
        emb_txt = self.encode_text(texts_) if texts_ else torch.zeros((0, self.emb_dim)).to(self.device)
        emb_img = self.encode_image(images_) if images_ else torch.zeros((0, self.emb_dim)).to(self.device)
        if emb_txt.device != emb_img.device:
            emb_txt = emb_txt.cpu()
            emb_img = emb_img.cpu()
        embs = torch.cat([emb_txt, emb_img], dim=0).cpu().numpy().astype(np.float32)
        embs = embs[idxs_]
        return embs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.
        Args:
            text: Image filename or Text to embed.
        Returns:
            Embedding.
        """
        return self.embed_documents([text])[0]


    def encode_text(self, texts):
        """
        Works with both:
        - OpenCLIP tokenizers that return a LongTensor [B, ctx_len]
        - HF-backed tokenizers that return a dict of tensors
        """
        with torch.no_grad():
            if self.backend == "oc":
                toks = self.text_tokenizer(texts)
                if isinstance(toks, dict):
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    tfeat = self.model.encode_text(**toks)
                else:
                    tfeat = self.model.encode_text(toks.to(self.device))
                
            elif self.backend == "hf":
                if "paligemma" in self.hf_model_id:
                    inputs = self.preprocess.tokenizer([text + self.vlm_text_prompts for text in texts], return_tensors="pt", padding=True).to(self.model.device)
                    if self.vlm_layer == -1:
                        # Image Encoder + Projection
                        input_ids = inputs["input_ids"].to(self.model.device)
                        emb_layer = self.model.get_input_embeddings()
                        tfeat = emb_layer(input_ids).mean(dim=1).to(torch.float32)

                    elif self.vlm_layer == -2:
                        raise ValueError("vlm_layer cannot be -2 when using text query!") 

                    elif 7 > self.vlm_layer >= 0:
                        # pick The gemma state (best is the 1st layer)
                        hid_sts = self.model(**inputs, output_hidden_states=True, return_dict=True)["hidden_states"]
                        tfeat = hid_sts[self.vlm_layer].mean(dim=1).to(torch.float32)
                    elif self.vlm_layer == 7:
                        # pick pre-logits
                        resp = self.model(**inputs, output_hidden_states=True, return_dict=True)["logits"].mean(dim=1)
                        tfeat = resp.to(torch.float32)
                elif "Qwen" in self.hf_model_id:
                    messages = [
                        [{
                            "role": "user",
                                "content": [
                                    {"type": "text", "text": text + self.vlm_text_prompts},
                                ],
                        }] for text in texts
                    ]
                    texts_processed = [
                        self.preprocess.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                        for msg in messages
                    ]
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.preprocess(
                        text=texts_processed,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                    resp = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    if "32B" in self.hf_model_id:
                        upper = 65
                    elif "3B" in self.hf_model_id:
                        upper = 37
                    else:
                        raise ValueError
                    if upper > self.vlm_layer >= 0:
                        hid_sts = resp["hidden_states"]
                        tfeat = hid_sts[self.vlm_layer].mean(dim=1).to(torch.float32)
                    elif self.vlm_layer == upper:
                        # pre-logits
                        tfeat = resp["logits"].mean(dim=1).to(torch.float32)

                    else:
                        raise ValueError
                    
                elif "dinov3" in self.hf_model_id:
                    raise ValueError("Text query not supported!")
                    
                elif "QQMM" in self.hf_model_id:
                    txt_feat = torch.cat([self.model.chat(text='Represent the following answer to an image classification task: ' + text) for text in texts], dim=0).to(torch.float32)
                    tfeat = txt_feat
                else:
                    raise NotImplementedError
            elif self.backend == "ol":
                # online models
                all_feats = []
                for text in texts:
                    if "seed" in self.online_model_nickname: 
                        img_feat = self.model.multimodal_embeddings.create(
                                    model=MODEL_NAME_MAPPING[self.online_model_nickname],
                                    input=[
                                        {
                                            "type":"text",
                                            "text": "Imagine the scene with the following object or person: " + text
                                        }
                                    ]
                                ).data.embedding
                    elif "google" in self.online_model_nickname:
                        img_feat = self.model.get_embeddings(image=None, contextual_text=text).text_embedding
                    else:
                        raise ValueError(f"Unknown model {self.online_model_nickname}")

                    this_feat = F.normalize(torch.tensor([img_feat]), dim=-1)
                    all_feats.append(this_feat)
                tfeat = torch.cat(all_feats, dim=0).cuda()

            else:
                raise NotImplementedError

            tfeat = F.normalize(tfeat, dim=-1)        
            return tfeat


    def encode_image(self, frame_paths):
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(frame_paths), self.batch_size), desc="Embed images", leave=False):
                batch_paths = frame_paths[i : i + self.batch_size]
                if self.backend == "oc":
                    ims = [self.preprocess(load_image_from_pkl_or_path(p)) for p in batch_paths]
                    ims = torch.stack(ims, dim=0).to(self.device)
                    img_feat = self.model.encode_image(ims)
                    
                elif self.backend == "hf":
                    if "paligemma" in self.hf_model_id:
                        if self.vlm_layer == -1:
                            # Image Encoder + Projection
                            ims = [self.preprocess(text="<image> " + self.vlm_image_prompts, images=load_image(p), return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)["pixel_values"] for p in batch_paths]
                            ims = torch.cat(ims, dim=0)
                            img_feat = self.model.get_image_features(ims).mean(dim=1)
                        elif self.vlm_layer == -2:
                            # Only Image Encoder
                            ims = [self.preprocess(text="<image> " + self.vlm_image_prompts, images=load_image(p), return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)["pixel_values"] for p in batch_paths]
                            ims = torch.cat(ims, dim=0)
                            image_outputs = self.model.vision_tower(ims)
                            img_feat = image_outputs.last_hidden_state.mean(dim=1)

                        elif 7 > self.vlm_layer >= 0:
                            # pick The gemma state (best is the 1st layer)
                            ims = [load_image(p) for p in batch_paths]
                            
                            inputs = self.preprocess(text=["<image> " + self.vlm_image_prompts] * len(ims), images=ims, return_tensors="pt", padding=True).to(self.model.device, dtype=torch.bfloat16)
                            hid_sts = self.model(**inputs, output_hidden_states=True, return_dict=True)["hidden_states"]
                            img_feat = hid_sts[self.vlm_layer].mean(dim=1)

                        elif self.vlm_layer == 7:
                            # pre-logits
                            ims = [load_image(p) for p in batch_paths]
                            inputs = self.preprocess(text=["<image> " + self.vlm_image_prompts] * len(ims), images=ims, return_tensors="pt", padding=True).to(self.model.device, dtype=torch.bfloat16)
                            img_feat = self.model(**inputs, output_hidden_states=True, return_dict=True)["logits"].mean(dim=1)

                    elif "Qwen" in self.hf_model_id:
                        messages = [
                            [{
                                "role": "user",
                                    "content": [
                                        {"type": "image", "image": p},
                                        {"type": "text", "text": self.vlm_image_prompts},
                                    ],
                            }] for p in batch_paths
                        ]
                        texts = [
                            self.preprocess.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                            for msg in messages
                        ]
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.preprocess(
                            text=texts,
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(self.model.device)
                        resp = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        # 3b-37 layers / 32b-65 layers
                        if "32B" in self.hf_model_id:
                            upper = 65
                        elif "3B" in self.hf_model_id:
                            upper = 37
                        else:
                            raise ValueError
                        if upper > self.vlm_layer >= 0:
                            hid_sts = resp["hidden_states"]
                            img_feat = hid_sts[self.vlm_layer].mean(dim=1).to(torch.float32)
                        elif self.vlm_layer == upper:
                            img_feat = resp["logits"].mean(dim=1).to(torch.float32)
                    
                    elif "dinov3" in self.hf_model_id:
                        images = [load_image(p) for p in batch_paths]
                        cls_token_id = 0
                        img_feat = torch.tensor(self.model(images))[:, 0, cls_token_id, :].to(torch.float32)
                    
                    elif "QQMM" in self.hf_model_id:
                        images = [Image.open(p).convert("RGB") for p in batch_paths]
                        img_feat = torch.cat([self.model.chat(text='Represent the given image for classification.', image=[image]) for image in images], dim=0)
                        img_feat = img_feat.to(torch.float32)
                        # 3.8k feature dim
                    else:
                        raise NotImplementedError
                    
                elif self.backend == "ol":
                    embs__ = []
                    for p in batch_paths:
                        if "seed" in self.online_model_nickname:
                            feat = self.model.multimodal_embeddings.create(
                                model=MODEL_NAME_MAPPING[self.online_model_nickname],
                                input=[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": file_to_data_url(p)
                                        }
                                    }
                                ]
                            ).data.embedding
                        elif "google" in self.online_model_nickname:
                            image = VImage.load_from_file(p)
                            feat = self.model.get_embeddings(image=image, contextual_text="Think about the scene:").image_embedding
                        else:
                            raise ValueError(f"Unknown model {self.online_model_nickname}")
                        feat = F.normalize(torch.tensor([feat]), dim=-1)  # (1, D)
                        embs__.append(feat)
                    if len(embs__) == 0:
                        embs__ = np.zeros((0, self.emb_dim), dtype=np.float32)
                    img_feat = torch.cat(embs__, dim=0).to(torch.float32).cuda()
                else:
                    raise NotImplementedError
                img_feat = F.normalize(img_feat, dim=-1)
                embs.append(img_feat.to(torch.float32))

        if len(embs) == 0:
            # Try to infer D robustly
            d = getattr(self.model, "text_projection", None)
            dim = d.shape[1] if d is not None else 512
            return np.zeros((0, dim), dtype=np.float32), []
        embs = torch.cat(embs, dim=0).to(torch.float32)
        return embs