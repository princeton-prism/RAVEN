# # import process dependencies
import argparse
# import pic/json dependencies
import base64, random
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
import pickle as pkl
import json
# import time/process/io dependencies
import sys
import glob
import tqdm
from io import BytesIO
import os
import time

# import text embedding
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from openai.types.responses import ResponseInputParam

# load this directory   
sys.path.append(sys.path[0] + '/..')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class OpenAICaptioner:
    def __init__(self, args):
        import openai
        self.client = openai.OpenAI(api_key=args.openai_api_key)
        self.model = args.openai_model
        self.query = args.query
        self.max_tokens = args.max_new_tokens
        self.temperature = args.temperature
        
    @staticmethod
    def _prep_image(image, max_side = 640, jpeg_quality = 80):
        """Prepare image for OpenAI API"""
        img = image.copy()
        w, h = img.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.Resampling.BILINEAR)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=jpeg_quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def _call_with_backoff(self, content, max_retries=8, base_delay=0.8):
        for attempt in range(1, max_retries + 1):
            try:
                input_messages: ResponseInputParam = [{"role": "user", "content": content}]
                resp = self.client.responses.create(
                    model=self.model,
                    input=input_messages,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp.output_text
            except openai.RateLimitError as e:
                # use the waiting time from the server
                retry_after = 1.0
                try:
                    retry_after = float(getattr(e, "response", None).headers.get("retry-after", retry_after))
                except Exception:
                    pass
                # add exponential backoff + jitter
                sleep_s = max(retry_after, base_delay * (2 ** (attempt - 1))) * (1 + 0.2 * random.random())
                print(f"[429] Rate limited. Sleeping {sleep_s:.2f}s before retry {attempt}/{max_retries}.")
                time.sleep(sleep_s)
            except openai.APIStatusError as e:
                if e.status_code in (500, 502, 503, 504):
                    sleep_s = base_delay * (2 ** (attempt - 1)) * (1 + 0.2 * random.random())
                    print(f"[{e.status_code}] Server error. Sleeping {sleep_s:.2f}s before retry {attempt}/{max_retries}.")
                    time.sleep(sleep_s)
                else:
                    raise
        raise RuntimeError("Exceeded max_retries for OpenAI call")

    def caption(self, images):
        content = [{"type": "input_text", "text": self.query}]
        for img in images:
            content.append({"type": "input_image", "image_url": self._prep_image(img, max_side=640, jpeg_quality=80)})

        return self._call_with_backoff(content)

def run_video_in_segs(args):

    SEQUENCE_ID=args.seq_id

    # load folders
    pkl_files = glob.glob(os.path.join(args.data_path, str(SEQUENCE_ID), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    time_start = times[0]
    for t, file in zip(times, pkl_files):
        if t - time_start > args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    
    # Initialize captioner based on selected model
    if args.captioner_type == "openai":
        captioner = OpenAICaptioner(args)
    else:
        raise ValueError(f"Unsupported captioner type: {args.captioner_type}")

    # if exists, then exit
    # captions_location = f'./data/{SEQUENCE_ID}/captions'
    captions_location = args.out_path
    if os.path.exists(captions_location) and not args.overwrite:
        print(f"Output directory {captions_location} already exists. Use --overwrite to overwrite.")
        exit()
        # shutil.rmtree(captions_location, ignore_errors=True)
    os.makedirs(captions_location, exist_ok=True)

    outputs = []

    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):

        images = []
        # depth = []
        # bboxes = []
        position = []
        rotation = []
        timestamp = []


        for file in file_names:
            with open(file, 'rb') as f:
                data = pkl.load(f)
                data['cam0'] = data['cam0'][:, :, ::-1]

                images.append(Image.fromarray(data['cam0'].astype('uint8'), 'RGB'))
                # depth.append(data['stereo'])
                # bboxes.append(data['bbox_3d'])
                position.append(data['position'])
                rotation.append(data['rotation'])
                timestamp.append(data['timestamp'])

        
        position = np.array(position)
        rotation = np.array(rotation)
        rotation = Rotation.from_quat(rotation).as_euler('xyz', degrees=True)
        timestamp = np.array(timestamp)

        # let's sample the images down to args.num_video_frames
        images = images[::30//args.num_video_frames]

        print(f"Processing segment {i+1}/{len(segments)}...")
        out_text = captioner.caption(images)

        print(f"Caption: {out_text[:1000]}..." if len(out_text) > 1000 else f"Caption: {out_text}")

        filename_start = os.path.basename(file_names[0])
        filename_end = os.path.basename(file_names[-1])


        text_embedding = embedder.embed_query(out_text)

        
        entity = {
            'id': file_names[0],
            'position': position.mean(axis=0),
            'theta': 3.14, # TEMPORARY: We are not using rotation information yet, so just leaving a placeholder
            'time': timestamp.mean(),
            'caption': out_text,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding
        }

        outputs.append(entity)

        # Add delay to respect API rate limits
        if args.api_delay > 0:
            time.sleep(args.api_delay)

    # Save outputs to JSON
    if args.captioner_type == "openai":
        output_filename = f'captions_{args.openai_model}_{args.seconds_per_caption}_secs.json'
    # You can add more captioner_type options, such as qwen, claude, etc.
    else:
        raise ValueError(f"Unsupported captioner type: {args.captioner_type}")
        
    output_path = os.path.join(captions_location, output_filename)

    # now save the outputs into a json
    with open(output_path, 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Processed {len(segments)} segments")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":

    default_query = "<video>\n You are a wandering around a university campus.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)

    # Captioner setting
    parser.add_argument("--captioner_type", type=str, default="openai",
                        choices=["openai"],
                        help="Type of captioner to use")

    # OpenAI API settings
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--openai_model", type=str, default="gpt-4-vision-preview",
                        help="OpenAI model to use")
    parser.add_argument("--api_delay", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    
    # bool
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory")

    # Image sampling
    parser.add_argument("--num_video_frames", type=int, default=6)
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    
    args = parser.parse_args()
    
    # set gpt-4o APT_KEY
    if args.captioner_type == "openai" and not args.openai_api_key:
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not args.openai_api_key:
            print("Error: OpenAI API key not found. Set --openai_api_key or OPENAI_API_KEY environment variable.")
            sys.exit(1)

    run_video_in_segs(args)