import argparse
import re
from io import BytesIO
import os, os.path as osp
import base64
import time

import requests
from PIL import Image
import numpy as np
import sys

import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json
import bisect
import tqdm
try:
    import anthropic
except ImportError:
    print("Anthropic package not found. Please install it via `pip install anthropic` if you plan to use Claude captioner.")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ClaudeCaptioner:
    def __init__(self, args):
        self.client = anthropic.Anthropic(
            api_key=args.claude_api_key  # You'll need to set this
        )
        self.model = args.claude_model
        self.query = args.query
        self.max_tokens = args.max_new_tokens
        self.temperature = args.temperature
        
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def caption(self, images):
        """Generate caption for a list of images using Claude"""
        try:
            # Prepare image content for Claude API
            content = []
            
            # Add the query text
            content.append({
                "type": "text",
                "text": self.query
            })
            
            # Add all images
            for i, image in enumerate(images):
                base64_image = self.image_to_base64(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            
            # Make API call to Claude
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"Error: Could not generate caption - {str(e)}"


class OpenAICaptioner:
    def __init__(self, args):
        import openai
        self.client = openai.OpenAI(api_key=args.openai_api_key)
        self.model = args.openai_model
        self.query = args.query
        self.max_tokens = args.max_new_tokens
        self.temperature = args.temperature
        
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        # show the image
        # image.show()
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    
    def caption(self, images):
        """Generate caption for a list of images using OpenAI GPT-4V"""
        attempt = 0
        waitt = 30
        while attempt < 3:
            attempt += 1
            try:
                # Prepare content for OpenAI API
                content = [{"type": "input_text", "text": self.query}]
                
                # Add all images
                for image in images:
                    base64_image = self.image_to_base64(image) # f"data:image/jpeg;base64,{img_str}"
                    # print(base64_image[:3000])  # Print the beginning of the base64 string for verification
                    content.append({
                        "type": "input_image",
                        "image_url": base64_image
                    })
                
                # Make API call to OpenAI
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                
                return response.output_text
                
            except Exception as e:
                print(f"Error generating caption: {e} - Attempt {attempt}/3")
                if attempt >= 3:
                    break
                print(f"Retrying...Waiting for {waitt} seconds before retrying.")
                time.sleep(waitt)  # Wait before retrying
                # return f"Error: Could not generate caption - {str(e)}"
        return "Error: Failed to generate caption after multiple attempts."
    

def run_video_in_segs(args):
    # Load folders
    pkl_files = glob.glob(os.path.join(args.input_dir, '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    starting_idx = bisect.bisect_right(times, args.starting_time_in_sec * 1e6)
    ending_idx = bisect.bisect_right(times, args.ending_time_in_sec * 1e6) if args.ending_time_in_sec > 0 else len(times)
    if len(times) > starting_idx >= 0:
        time_start = times[starting_idx]
    else:
        raise ValueError(f"starting_time_in_sec {args.starting_time_in_sec} is out of bounds.") 
    
    for t, file in zip(times[starting_idx:ending_idx], pkl_files[starting_idx:ending_idx]):
        if t - time_start > args.seconds_per_caption * 1e6:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    # Initialize embedder
    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    
    # Initialize captioner based on selected model
    if args.captioner_type == "claude":
        captioner = ClaudeCaptioner(args)
    elif args.captioner_type == "openai":
        captioner = OpenAICaptioner(args)
    else:
        raise ValueError(f"Unsupported captioner type: {args.captioner_type}")

    # Check if output already exists
    captions_location = args.output_dir
    if os.path.exists(captions_location) and not args.overwrite:
        print(f"Output directory {captions_location} already exists. Use --overwrite to overwrite.")
        exit()
    os.makedirs(captions_location, exist_ok=True)

    outputs = []

    print(f"Processing {len(segments)} segments...")
    
    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):
        images = []
        position = []
        timestamp = []

        # Load images from pickle files
        for file in file_names:
            with open(file, 'rb') as f:
                data = pkl.load(f)
                # Convert BGR to RGB for PIL
                data['cam0'] = data['cam0'][:, :, ::-1]

                images.append(PILImage.fromarray(data['cam0'].astype('uint8'), 'RGB'))
                position.append(data['position'])
                timestamp.append(data['timestamp'])

        timestamp = np.array(timestamp)

        # Sample images down to args.num_video_frames
        if len(images) > args.num_video_frames:
            step = len(images) // args.num_video_frames
            images = images[::step]

        # Generate caption
        print(f"Processing segment {i+1}/{len(segments)}...")
        out_text = captioner.caption(images)
        print(f"Caption: {out_text[:1000]}..." if len(out_text) > 1000 else f"Caption: {out_text}")

        filename_start = os.path.basename(file_names[0])
        filename_end = os.path.basename(file_names[-1])

        # Generate text embedding
        text_embedding = embedder.embed_query(out_text)
        entity = {
            'id': file_names[0],
            'position': position,
            'time': np.array([float(xx) for xx in timestamp]).mean(),
            'caption': out_text,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding,
            'segment_index': i,
            'num_frames': len(file_names),
            'sampled_frames': len(images)
        }

        outputs.append(entity)

        # Add delay to respect API rate limits
        if args.api_delay > 0:
            time.sleep(args.api_delay)

    # Save outputs to JSON
    output_filename = f'captions_{args.openai_model if args.captioner_type == "openai" else args.claude_model}_{args.seconds_per_caption}_secs.json'
    output_path = os.path.join(captions_location, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder, indent=2)

    print(f"\nProcessing completed!")
    print(f"Processed {len(segments)} segments")
    print(f"Output saved to: {output_path}")


def run_images(input_dir: str, output_dir: str):
    """
    Caption every single image under args.input_dir (recursively).
    For each image:
    - caption with the selected captioner
    - position (x, y, z)
    - file_start == file_end == image_file
    Results are saved as a single JSON file under args.output_dir.
    """
    pkl_files = glob.glob(os.path.join(input_dir, '*.pkl'))
    # import ipdb; ipdb.set_trace()
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][-7:-4])) # the -7:-4 is a experimental number, remove the '.pkl' extension
                                                                 # The numbers can change to different file name

    # import ipdb; ipdb.set_trace()
    
    # Initialize embedder
    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

    # Initialize captioner
    if args.captioner_type == "claude":
        captioner = ClaudeCaptioner(args)
    elif args.captioner_type == "openai":
        captioner = OpenAICaptioner(args)
    else:
        raise ValueError(f"Unsupported captioner type: {args.captioner_type}")

    # prepare output dir / file
    # Check if output already exists
    captions_location = output_dir
    # if os.path.exists(captions_location) and not args.overwrite:
    #     print(f"Output directory {captions_location} already exists. Use --overwrite to overwrite.")
    #     return False
    os.makedirs(captions_location, exist_ok=True)
    
    # prepare output file
    perfix = f"{args.output_dir.split('/')[-2]}_{args.output_dir.split('/')[-1]}"
    out_fname = f"image_captions_{perfix}_{args.openai_model if args.captioner_type=='openai' else args.claude_model}.json"
    out_path = os.path.join(captions_location, out_fname)

    outputs = []
    print(f"Found {len(pkl_files)} images. Captioning...")

    for i, img_path in tqdm.tqdm(enumerate(pkl_files), total=len(pkl_files)):
        # load image → PIL
        try:
            with open(img_path, 'rb') as f:
                data = pkl.load(f)
                # Convert BGR to RGB for PIL
                data['cam0'] = data['cam0'][:, :, ::-1]
                im = PILImage.fromarray(data['cam0'].astype('uint8'), 'RGB')
                im = im.convert("RGB")
        except Exception as e:
            print(f"Skip unreadable image: {img_path} ({e})")
            continue

        # caption (single image list to reuse your captioner API)
        cap_text = captioner.caption([im])
        if isinstance(cap_text, str) and len(cap_text) > 1000:
            print(f"[{i+1}/{len(pkl_files)}] {os.path.basename(img_path)} -> {cap_text[:1000]}...")
        else:
            print(f"[{i+1}/{len(pkl_files)}] {os.path.basename(img_path)} -> {cap_text}")

        # embed text
        text_emb = embedder.embed_query(cap_text)

        # time (best-effort): try number from name; else use index
        name = os.path.basename(img_path)
        m = re.findall(r"\d+", name)
        t_val = float(m[0]) if m else float(i)

        entity = {
            "id": img_path,
            "position": [0, 0, 0],             # ← per your requirement
            "time": t_val,                     # optional, kept for schema symmetry
            "caption": cap_text,
            "file_start": name,                # start_frame = this frame
            "file_end": name,                  # end_frame   = this frame
            "text_embedding": text_emb,
            "segment_index": i,
            "num_frames": 1,
            "sampled_frames": 1
        }
        outputs.append(entity)

        if args.api_delay > 0:
            time.sleep(args.api_delay)

    # save JSON
    with open(out_path, "w") as f:
        json.dump(outputs, f, cls=NumpyEncoder, indent=2)

    print("\nImage captioning completed!")
    print(f"Processed {len(outputs)} images")
    print(f"Output saved to: {out_path}")


def recursive_run_images(args):
    """
    Recursively caption every single image under args.input_dir (recursively).
    Results are saved as a single JSON file under args.output_dir.
    """
    if not hasattr(args, "input_dir") or not hasattr(args, "output_dir"):
        raise ValueError("Args must have 'input_dir' and 'output_dir' attributes.")

    in_root = os.path.abspath(args.input_dir)
    out_root = os.path.abspath(args.output_dir)

    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {in_root}")

    found_any = False
    processed = 0
    # import ipdb; ipdb.set_trace()
    for root, dirs, files in os.walk(in_root):
        # check if the directory name is "frames"
        if os.path.basename(root).lower() != "frames":
            continue

        # check if the directory has .pkl files
        has_pkl = any(f.lower().endswith(".pkl") for f in files)
        if not has_pkl:
            continue

        found_any = True

        # get the parent directory
        parent_dir = os.path.dirname(root)
        # get the relative path of the parent directory
        rel_parent = os.path.relpath(parent_dir, in_root)

        # get the output directory
        mapped_out_dir = os.path.join(out_root, rel_parent)

        # run_images will exit if the output directory already exists and --overwrite is not set,
        if os.path.exists(mapped_out_dir) and not args.overwrite:
            print(f"[recursive] Output directory {mapped_out_dir} already exists. Use --overwrite to overwrite.")
            continue

        os.makedirs(mapped_out_dir, exist_ok=True)

        print("\n[recursive] Processing:")
        print(f"  frames_dir : {root}")
        print(f"  parent_dir : {parent_dir}")
        print(f"  out_dir    : {mapped_out_dir}")

        # call the existing image captioning logic
        try:
            run_images(root, mapped_out_dir)
        except Exception as e:
            print(f"[recursive] Error processing '{root}': {e}")
            continue
        processed += 1

    if not found_any:
        raise ValueError(f"No 'frames' directories with .pkl files found under '{in_root}'")

    print(f"\n[recursive] Done. Processed {processed} 'frames' directories.")

if __name__ == "__main__":
    default_query = """
You are wandering around a university campus. Please describe in detail what you see in the few seconds of the video. 
Specifically focus on the people, objects, environmental features, events/activities, and other interesting details. 
Think step by step about these details and be very specific."""

    parser = argparse.ArgumentParser()
    
    # Captioner selection
    parser.add_argument("--captioner_type", type=str, default="claude", 
                        choices=["claude", "openai"],
                        help="Type of captioner to use")
    
    # Claude API settings
    parser.add_argument("--claude_api_key", type=str, default=None,
                        help="Claude API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--claude_model", type=str, default="claude-3-sonnet-20240229",
                        choices=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                        help="Claude model to use")
    
    # OpenAI API settings (as fallback option)
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--openai_model", type=str, default="gpt-4-vision-preview",
                        help="OpenAI model to use")
    
    # General settings
    parser.add_argument("--input_dir", type=str, default="./coda_data")
    parser.add_argument("--output_dir", type=str, default="./data/captions")
    parser.add_argument("--starting_time_in_sec", type=int, default=0)
    parser.add_argument("--ending_time_in_sec", type=int, default=0)
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument("--num_video_frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=5120)
    parser.add_argument("--api_delay", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    
    # Image mode
    parser.add_argument("--from_images", action="store_true",
                        help="Run in image mode")
    
    args = parser.parse_args()

    # Set API keys from environment if not provided
    if args.captioner_type == "claude" and not args.claude_api_key:
        args.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not args.claude_api_key:
            print("Error: Claude API key not found. Set --claude_api_key or ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
    
    if args.captioner_type == "openai" and not args.openai_api_key:
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not args.openai_api_key:
            print("Error: OpenAI API key not found. Set --openai_api_key or OPENAI_API_KEY environment variable.")
            sys.exit(1)

    if args.from_images:
        recursive_run_images(args)
    else:
        run_video_in_segs(args)
