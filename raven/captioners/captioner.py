from PIL import Image
from io import BytesIO
import base64
import time

class Captioner:
    def caption(self, images: list[Image.Image]):
        raise NotImplementedError
    

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
        return "Error: Failed to generate caption after multiple attempts."
    