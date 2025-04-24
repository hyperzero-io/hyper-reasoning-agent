import io
import logging
from uuid import uuid4
from pathlib import Path
import base64
from agno.playground import Playground, serve_playground_app
from config import config
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat, OpenAILike
from agno.models.azure import AzureOpenAI
from agno.models.google import Gemini
from agno.tools import Toolkit
from agno.tools.sleep import SleepTools
from agno.tools.reasoning import ReasoningTools
from agno.media import Image
from PIL import Image as PILImage, ImageDraw, ImageEnhance, ImageFont, ImageFilter
from typing import List
import numpy as np
from openai import OpenAI
from typing import Iterator, Union
import requests

logger = logging.getLogger(__name__)
import json
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")



class OnlineResearchToolkit(Toolkit):
    """Toolkit for performing online research tasks, starting with Google Grounding."""
    def __init__(self):
        # Call the base class __init__
        super().__init__(name="online_research_toolkit") 
        
        # Register functions (using the base class method)
        self.register(self.google_grounding_research)
        logger.info("OnlineResearchToolkit initialized.")

    def google_grounding_research(
        self,
        agent: Agent, # The agent calling the tool (not directly used for model call here)
        question: str,
        image_path: str = None,
    ) -> str:
        """
        Answers a research question using Google Gemini with grounding based on the provided context.

        Args:
            agent (Agent): The agent instance running the tool.
            question (str): The research question to answer.

        Returns:
            str: JSON string containing 'research_answer' and 'sources' (list of dicts with url/title).
        """
        gemini_api_key = config.GEMINI_API_KEY
        if not gemini_api_key:
            logger.error("GOOGLE_API_KEY_GEMINI_SHARED environment variable not set.")
            return json.dumps({"success": False, "error": "API key not configured for research tool."})

        logger.info(f"Performing Google Grounding research for question: '{question[:50]}...'")
        
        try:
            # Instantiate Gemini specifically for this grounding task
            # Using a recent model, adjust if needed
            grounding_model = Gemini(
                id="gemini-2.5-flash-preview-04-17", 
                api_key=gemini_api_key,
                grounding=True,
                # Add other params if needed (e.g., vertexai, project_id, location)
            )
            
            # Create a temporary agent instance for this specific task
            research_agent = Agent(
                model=grounding_model,
                # Simple instructions focused on the task
                instructions="You are a research assistant. Answer the user's question based *only* on the provided context. Cite sources from the context.",
            )
            
            # Construct the prompt
            prompt = f"Question: {question}"
            images = None
            if image_path:
                try:
                    with open(image_path, "rb") as img_file:
                        image_bytes = img_file.read()
                    images = [Image(content=image_bytes, format="jpeg", name="uploaded_image.jpeg")]
                except Exception as e:
                    logger.warning(f"Could not read image at {image_path}: {e}")
            
            # Call Gemini asynchronously
            if images:
                response: RunResponse = research_agent.run(prompt, images=images)
            else:
                response: RunResponse = research_agent.run(prompt)
            
            research_answer = response.content
            sources = []
            
            # Check for citations in the response
            if hasattr(response, 'citations') and response.citations and hasattr(response.citations, 'urls') and response.citations.urls:
                logger.info(f"Found {len(response.citations.urls)} citations in grounding response.")
                for i, citation in enumerate(response.citations.urls):
                    source_url = getattr(citation, 'url', None)
                    source_title = getattr(citation, 'title', f"Source {i+1}")
                    if source_url:
                        sources.append({
                            "index": i + 1,
                            "url": source_url,
                            "title": source_title
                        })
            else:
                logger.info("No citations found in grounding response.")

            result_dict = {
                "success": True, 
                "research_answer": research_answer,
                "sources": sources
            }
            return json.dumps(result_dict)

        except Exception as e:
            logger.error(f"Error during Google Grounding research: {e}", exc_info=True)
            # Return error as JSON string
            return json.dumps({'success': False, 'error': str(e)})

class HyperReasoningToolkit(Toolkit):
    """
    Comprehensive image-processing toolkit for LLM agents:
      - preprocess_for_better_llm: enhancement + grid
      - crop: rectangular region extraction
      - draw_marker: annotate coords
      - rotate, resize, grayscale conversion
      - adjust_contrast, adjust_brightness, threshold, detect_edges, overlay_image
      - llm_image_analyze: ask LLM to describe/analyze image content
      - llm_image_verification: verify hypotheses about image with LLM
    """
    def __init__(self, **kwargs):
        super().__init__(name="hyper_reasoning", **kwargs)
        # Register all image-processing and LLM tools
        for fn in [
            self.crop, self.draw_marker, self.add_text,
            self.llm_image_analyze, self.llm_image_verification, self.llm_hyperintelligent_thinker, self.llm_hyperlogical_thinker,
            self.save_image_for_processing,
            self.crop_and_zoom,
            self.adjust_brightness, self.adjust_contrast, self.detect_edges, self.convert_to_grayscale,
            self.draw_rectangle, self.draw_circle, self.draw_line,
            self.draw_polygon, self.draw_arrow, self.draw_filled_rectangle, self.blur_region, self.add_watermark,
            self.draw_filled_circle, self.invert_colors, self.histogram_equalization, self.auto_crop, self.mosaic_region, self.overlay_text_box
        ]:
            self.register(fn)

    def save_image_for_processing(self, agent: Agent):
        """
        Save all images from agent.images (list of Image objects) to local files.
        Handles both Image.content (bytes) and Image.url (download) cases.
        Returns a list of local file paths for reference.
        """

        saved_images = []
        user_imgs = agent.run_messages.user_message.images or []
        # print(user_imgs)
        save_dir = Path("saved_images")
        save_dir.mkdir(exist_ok=True)
        if user_imgs and isinstance(user_imgs, list):
            for idx, img in enumerate(user_imgs):
                # Determine filename
                ext = "jpg"
                if hasattr(img, 'name') and img.name:
                    ext = os.path.splitext(img.name)[-1][1:] or "jpg"
                fname = f"img_{uuid4().hex}.{ext}"
                fpath = save_dir / fname
                # Save from content (bytes)
                with open(fpath, "wb") as f:
                    if isinstance(img, dict):
                        if 'content' in img:
                            data = img['content']
                        elif 'url' in img:
                            response = requests.get(img['url'])
                            data = response.content
                        else:
                            logger.warning("Image dict has no 'content' or 'url'")
                            continue
                    elif hasattr(img, 'content'):
                        data = img.content
                    elif isinstance(img, bytes):
                        data = img
                    elif hasattr(img, 'url'):
                        response = requests.get(img.url)
                        data = response.content
                    else:
                        logger.warning(f"Unsupported image type: {type(img)}")
                        continue
                    f.write(data)
                try:
                    with PILImage.open(fpath) as pil_img:
                        pil_img.verify()
                    saved_images.append({"image_file_name": fname, "image_file_path": str(fpath)})
                except Exception as e:
                    logger.warning(f"Failed to save image content: {e}")
                    continue
        print(saved_images)
        if len(saved_images) > 0:
            return json.dumps(saved_images)
        else:
            return json.dumps([])

    def crop(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, output_path: str = None) -> str:
        """Crop a region from the image and save it. Adds 100px padding to all sides, clipped to image boundaries."""
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        padding = 100
        x_min_p = max(0, x_min - padding)
        y_min_p = max(0, y_min - padding)
        x_max_p = min(w, x_max + padding)
        y_max_p = min(h, y_max + padding)
        region = img.crop((x_min_p, y_min_p, x_max_p, y_max_p))
        out = output_path or f"crop_{uuid4().hex}.jpeg"
        region.save(out)
        return out

    def crop_and_zoom(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, zoom_factor: int = 2, padding: int = 100, output_path: str = None) -> str:
        """
        Crop a region from the image (with padding), zoom (resize) it by zoom_factor, and save the result.
        Args:
            image_path: Path to input image
            x_min, y_min, x_max, y_max: Crop box coordinates
            zoom_factor: How much to enlarge the cropped region (default=2)
            padding: Padding in pixels to add to all sides (default=100)
            output_path: Where to save the cropped & zoomed image (optional)
        Returns:
            Path to the saved cropped & zoomed image
        """
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        x_min_p = max(0, x_min - padding)
        y_min_p = max(0, y_min - padding)
        x_max_p = min(w, x_max + padding)
        y_max_p = min(h, y_max + padding)
        region = img.crop((x_min_p, y_min_p, x_max_p, y_max_p))
        # Zoom the cropped region
        region_zoomed = region.resize((region.width * zoom_factor, region.height * zoom_factor), PILImage.LANCZOS)
        out = output_path or f"crop_zoom_{uuid4().hex}.jpeg"
        region_zoomed.save(out)
        return out

    from typing import List

    def adjust_brightness(self, image_path: str, factor: float, output_path: str = None) -> str:
        """
        Adjust the brightness of an image.
        Args:
            image_path: Path to input image
            factor: Brightness factor (1.0 = original, <1.0 = darker, >1.0 = brighter)
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        from PIL import ImageEnhance
        img = PILImage.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        bright = enhancer.enhance(factor)
        out = output_path or f"bright_{uuid4().hex}.jpeg"
        bright.save(out)
        return out

    def adjust_contrast(self, image_path: str, factor: float, output_path: str = None) -> str:
        """
        Adjust the contrast of an image.
        Args:
            image_path: Path to input image
            factor: Contrast factor (1.0 = original, <1.0 = less, >1.0 = more)
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        from PIL import ImageEnhance
        img = PILImage.open(image_path)
        enhancer = ImageEnhance.Contrast(img)
        contrast = enhancer.enhance(factor)
        out = output_path or f"contrast_{uuid4().hex}.jpeg"
        contrast.save(out)
        return out

    def detect_edges(self, image_path: str, output_path: str = None) -> str:
        """
        Detect edges in an image using Pillow's FIND_EDGES filter.
        Args:
            image_path: Path to input image
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        img = PILImage.open(image_path)
        edges = img.filter(ImageFilter.FIND_EDGES)
        # Convert to RGB if image has alpha channel and saving as JPEG
        out = output_path or f"edges_{uuid4().hex}.jpeg"
        if out.lower().endswith('.jpg') or out.lower().endswith('.jpeg'):
            if edges.mode == 'RGBA' or edges.mode == 'LA':
                edges = edges.convert('RGB')
        edges.save(out)
        return out

    def convert_to_grayscale(self, image_path: str, output_path: str = None) -> str:
        """
        Convert an image to grayscale.
        Args:
            image_path: Path to input image
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        img = PILImage.open(image_path)
        gray = img.convert("L")
        out = output_path or f"gray_{uuid4().hex}.jpeg"
        gray.save(out)
        return out

    def draw_rectangle(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, outline: str = "red", width: int = 3, output_path: str = None) -> str:
        """
        Draw a rectangle on the image.
        Args:
            image_path: Path to input image
            x_min, y_min, x_max, y_max: Rectangle coordinates
            outline: Rectangle outline color
            width: Outline width
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline, width=width)
        out = output_path or f"rect_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def draw_circle(self, image_path: str, center_x: int, center_y: int, radius: int, outline: str = "blue", width: int = 3, output_path: str = None) -> str:
        """
        Draw a circle at a specified center and radius.
        Args:
            image_path: Path to input image
            center_x, center_y: Center of the circle
            radius: Circle radius
            outline: Circle outline color
            width: Outline width
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(bbox, outline=outline, width=width)
        out = output_path or f"circle_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def draw_line(self, image_path: str, x1: int, y1: int, x2: int, y2: int, fill: str = "green", width: int = 3, output_path: str = None) -> str:
        """
        Draw a line between two points.
        Args:
            image_path: Path to input image
            x1, y1: Start point
            x2, y2: End point
            fill: Line color
            width: Line width
            output_path: Where to save the result (optional)
        Returns: Path to the saved image
        """
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        out = output_path or f"line_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    from typing import List

    def draw_polygon(self, image_path: str, points: List[List[int]], outline: str = "yellow", width: int = 3, output_path: str = None) -> str:
        """
        Draw a polygon on the image.
        Args:
            points: List of [x, y] pairs (e.g., [[x1, y1], [x2, y2], ...]).
        """
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(pt) for pt in points], outline=outline, width=width)
        out = output_path or f"polygon_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def draw_arrow(self, image_path: str, x1: int, y1: int, x2: int, y2: int, fill: str = "orange", width: int = 3, output_path: str = None) -> str:
        import math
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
        # Draw arrowhead
        arrow_length = 15
        angle = math.atan2(y2 - y1, x2 - x1)
        for theta in [math.pi/8, -math.pi/8]:
            dx = int(arrow_length * math.cos(angle + theta))
            dy = int(arrow_length * math.sin(angle + theta))
            draw.line([(x2, y2), (x2 - dx, y2 - dy)], fill=fill, width=width)
        out = output_path or f"arrow_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def draw_filled_rectangle(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, fill: str = "red", output_path: str = None) -> str:
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([x_min, y_min, x_max, y_max], fill=fill)
        out = output_path or f"filled_rect_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def blur_region(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, radius: int = 10, output_path: str = None) -> str:
        img = PILImage.open(image_path).convert("RGB")
        region = img.crop((x_min, y_min, x_max, y_max)).filter(ImageFilter.GaussianBlur(radius))
        img.paste(region, (x_min, y_min))
        out = output_path or f"blur_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def add_watermark(self, image_path: str, text: str, position: List[int] = [10, 10], opacity: int = 128, output_path: str = None) -> str:
        """
        Add a semi-transparent text watermark.
        Args:
            position: [x, y] coordinates for the watermark.
        """
        img = PILImage.open(image_path).convert("RGBA")
        from PIL import ImageDraw, ImageFont
        watermark = PILImage.new("RGBA", img.size)
        draw = ImageDraw.Draw(watermark)
        font = ImageFont.load_default()
        draw.text(tuple(position), text, fill=(255, 255, 255, opacity), font=font)
        watermarked = PILImage.alpha_composite(img, watermark)
        out = output_path or f"watermark_{uuid4().hex}.png"
        watermarked.save(out)
        return out

    def draw_filled_circle(self, image_path: str, center_x: int, center_y: int, radius: int, fill: str = "blue", output_path: str = None) -> str:
        img = PILImage.open(image_path).convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(bbox, fill=fill)
        out = output_path or f"filled_circle_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def invert_colors(self, image_path: str, output_path: str = None) -> str:
        img = PILImage.open(image_path)
        from PIL import ImageOps
        inverted = ImageOps.invert(img.convert("RGB"))
        out = output_path or f"invert_{uuid4().hex}.jpeg"
        inverted.save(out)
        return out

    def histogram_equalization(self, image_path: str, output_path: str = None) -> str:
        img = PILImage.open(image_path).convert("L")
        from PIL import ImageOps
        eq = ImageOps.equalize(img)
        out = output_path or f"equalized_{uuid4().hex}.jpeg"
        eq.save(out)
        return out

    def auto_crop(self, image_path: str, bg_color: str = "white", output_path: str = None) -> str:
        img = PILImage.open(image_path)
        from PIL import ImageChops
        bg = PILImage.new(img.mode, img.size, bg_color)
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        cropped = img.crop(bbox) if bbox else img
        out = output_path or f"autocrop_{uuid4().hex}.jpeg"
        cropped.save(out)
        return out

    def mosaic_region(self, image_path: str, x_min: int, y_min: int, x_max: int, y_max: int, mosaic_size: int = 10, output_path: str = None) -> str:
        img = PILImage.open(image_path).convert("RGB")
        region = img.crop((x_min, y_min, x_max, y_max))
        region = region.resize((max(1, region.width // mosaic_size), max(1, region.height // mosaic_size)), PILImage.NEAREST)
        region = region.resize((x_max - x_min, y_max - y_min), PILImage.NEAREST)
        img.paste(region, (x_min, y_min))
        out = output_path or f"mosaic_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def overlay_text_box(self, image_path: str, text: str, box: List[int], fill: List[int] = [0,0,0,128], text_fill: List[int] = [255,255,255,255], output_path: str = None) -> str:
        """
        Draw a semi-transparent rectangle with text.
        Args:
            box: [x_min, y_min, x_max, y_max] coordinates for the rectangle.
            fill: [R, G, B, A] color for the rectangle.
            text_fill: [R, G, B, A] color for the text.
        """
        img = PILImage.open(image_path).convert("RGBA")
        from PIL import ImageDraw, ImageFont
        overlay = PILImage.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(box), fill=tuple(fill))
        font = ImageFont.load_default()
        text_pos = (box[0] + 5, box[1] + 5)
        draw.text(text_pos, text, fill=tuple(text_fill), font=font)
        combined = PILImage.alpha_composite(img, overlay)
        out = output_path or f"textbox_{uuid4().hex}.png"
        combined.save(out)
        return out

    def draw_marker(
        self,
        image_path: str,
        x: int,
        y: int,
        radius: int = 15,
        color: List[int] = [255, 0, 0],
        output_path: str = None,
    ) -> str:
        """
        Annotate the image with a circular marker at (x, y).
        Args:
            image_path (str): Path to the image file.
            x (int): X coordinate.
            y (int): Y coordinate.
            radius (int, optional): Marker radius. Defaults to 15.
            color (List[int], optional): RGB color as [R, G, B], must be a list of 3 integers (RGB). Defaults to [255, 0, 0].
            output_path (str, optional): Output file path. Defaults to None.
        Returns:
            str: Path to the annotated image.
        """
        if not (isinstance(color, list) and len(color) == 3):
            raise ValueError("color must be a list of 3 integers (RGB)")
        img = PILImage.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], outline=tuple(color), width=3)
        out = output_path or f"marker_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    def add_text(
        self,
        image_path: str,
        text: str,
        x: int,
        y: int,
        color: List[int] = [255, 0, 0],
        font_size: int = 32,
        output_path: str = None,
        font_path: str = None,
    ) -> str:
        """
        Draw text on the image at (x, y).
        Args:
            image_path (str): Path to the image file.
            text (str): Text to draw.
            x (int): X coordinate.
            y (int): Y coordinate.
            color (list, optional): RGB color. Defaults to [255, 0, 0].
            font_size (int, optional): Font size. Defaults to 32.
            output_path (str, optional): Output file path. Defaults to None.
            font_path (str, optional): Path to a .ttf font file. Defaults to None (uses default).
        Returns:
            str: Path to the annotated image.
        """
        img = PILImage.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Use Roboto-Regular.ttf as default
                roboto_path = str(Path(__file__).parent + "fonts" + "Roboto-Regular.ttf")
                font = ImageFont.truetype(roboto_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x, y), text, fill=tuple(color), font=font)
        out = output_path or f"text_{uuid4().hex}.jpeg"
        img.save(out)
        return out

    # ... other utility methods unchanged ...

    def llm_image_analyze(self, image_path: str, question: str) -> str:
        """Use LLM to analyze image content based on a question."""
        with open(image_path, 'rb') as f: data = f.read()
        b64 = base64.b64encode(data).decode()
        messages = [
            {"role": "system", "content": "You are a vision assistant. Describe or analyze the requested aspect of the image."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL_GPT4,
            messages=messages,
        )
        return response.choices[0].message.content

    def llm_hyperintelligent_thinker(self, question: str, context: str) -> str:
        """Use GPT-4.5-preview-2025-02-27 to answer a question with context (text only)."""
        messages = [
            {"role": "system", "content": "You are a hyperintelligent reasoning agent. Integrate the question and user context for the most insightful answer."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL_GPT45,
            messages=messages,
        )
        return response.choices[0].message.content

    def llm_hyperlogical_thinker(self, question: str, context: str) -> str:
        """Use o4-mini-2025-04-16 to answer a question with context, focusing on logical reasoning (text only)."""
        messages = [
            {"role": "system", "content": "You are a hyperlogical reasoning agent. Integrate the question and user context for the most rigorous logical answer."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL_O4MINI,
            messages=messages,
        )
        return response.choices[0].message.content
    

    def llm_image_verification(self, image_path: str, statement: str) -> str:
        """Verify or refute a statement about the image using LLM."""
        with open(image_path, 'rb') as f: data = f.read()
        b64 = base64.b64encode(data).decode()
        messages = [
            {"role": "system", "content": "You are a vision assistant. Confirm or deny the statement about the image with concise reasoning."},
            {"role": "user", "content": [
                {"type": "text", "text": statement},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        resp = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL_GPT4,
            messages=messages,
        )
        return resp.choices[0].message.content


def _load_font(size: int = 20):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

from textwrap import dedent
model = OpenAILike(
    id=config.OPENAI_MODEL_GPT4,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    # temperature=0.7,
)
hyper_tools = HyperReasoningToolkit()
tools = [
    ReasoningTools(think=True, analyze=True, add_instructions=True),
    SleepTools(),
    hyper_tools,
    OnlineResearchToolkit(),
]
agent = Agent(
    name="Hyper Reasoning Agent",
    model=model,
    tools=tools,
    instructions=dedent("""
        ## 🧠 Role: Hyper Reasoning Agent (Always using think and analyze tools)

        When I send a task (optionally with an image), follow this 9‑phase hyper reasoning workflow:

        ---

        ### 📦 Image Toolkit Usage Guide (All processed images should be saved into ./processed_images/ folder, and image file name should include uuid to avoid overwriting)
        Use these tools as needed to solve vision, annotation, and preprocessing tasks:
        - **crop / crop_and_zoom**: Use to isolate regions, especially for small/far objects (crop_and_zoom for extra detail).
        - **adjust_brightness / adjust_contrast**: Use if image is too dark, bright, or low/high contrast for analysis.
        - **detect_edges**: Use to highlight object boundaries or for edge-based reasoning.
        - **convert_to_grayscale**: Use to simplify analysis or before thresholding.
        - **draw_rectangle / draw_circle / draw_line**: Use for marking, highlighting, or annotating specific regions or features.
        - **draw_polygon**: Use for irregular region annotation.
        - **draw_arrow**: Use to indicate direction, flow, or relationships.
        - **draw_filled_rectangle / draw_filled_circle**: Use to mask, highlight, or anonymize regions.
        - **blur_region / mosaic_region**: Use to anonymize faces, text, or sensitive details.
        - **add_watermark / overlay_text_box**: Use to add labels, callouts, or provenance to images.
        - **invert_colors**: Use for contrast inversion or special effects.
        - **histogram_equalization**: Use to enhance contrast in poorly-lit or faded images.
        - **auto_crop**: Use to remove excess background or whitespace.
        - **llm_image_analyze**: Use to extract features (texture, shape, text, etc.) from the image.
        - **llm_image_verification**: Use to verify hypotheses about the image.
        
        ### Reasoning Tools
        - **think** and **analyze**: Always use these tools for planning and reasoning throughout the process.
        - **llm_hyperintelligent_thinker**: Use to generate deep multimodal insights combining image, context, and question using the GPT‑4.5 model (When need something creative, innovative, or imaginative)
        - **llm_hyperlogical_thinker**: Use to produce step‑by‑step logical reasoning and hypothesis testing using the o4‑mini model (When need something logical, analytical, or systematic)
        - **online_research_toolkit**: Use to search for information, context, or verification from the web (When need something external, verified, or up-to-date)

        ---

        ### 1. **Initial Understanding — Always Ask First**
        Always use save_image_for_processing to save image to local storage before any processing.
        - Step: Analyze both the image and the user's question together.
        - Goal: Form a **neutral, independent** understanding of the situation by considering both the visual content and the user's query before making any assumptions.
        - Output:
        - Scene layout, environment type (from image)
        - Notable visual elements (text, objects, flags, people, etc.) (from image)
        - Any patterns, actions, or symbols (from image)
        - Key aspects, intent, or requirements expressed in the user question
        - How the user question may relate to or focus attention on specific parts of the image or task

        ---

        ### 2. **Contextual Inference**
        - Suggest multiple **plausible scenarios** the image may represent.  
        - Use visual, cultural, geographic, and behavioral clues.  
        - Keep all reasoning hedged (e.g. "possibly", "appears to be") at this stage.  

        ---

        ### 3. **User Query Alignment**
        - Interpret the user's prompt **in context of the image**, not the other way around.  
        - Avoid assumptions based on the user's language — prioritize visual evidence.  

        ---

        ### 4. **Object-Level Reasoning**
        For each candidate region or object:  
        - `crop`: Isolate clearly.  
        - `crop_and_zoom`: Zoom in on the region of interest (when needed for far and small objects)
        - `llm_image_analyze`: Extract features (texture, shape, text, etc.)  
        - `llm_image_verification`: Form and test hypotheses.  
        - `google_grounding_research`:  
        - Use **keywords in the detected language** from the image  
        - Avoid translating to a fixed language unless necessary  

        ---

        ### 5. **Elimination Pass**
        - For each plausible scenario, ask:  
        - Does the evidence **refute** this possibility?  
        - Are key supporting features **missing or contradictory**?  
        - Actively **eliminate weak or unsupported scenarios**.  
        - Once eliminated, **do not return to that line of reasoning**.  
        - Repeat until **1 or 2 highest‑probability scenarios remain**.  

        ---

        ### 6. **Uncertainty Management**
        - For each insight or claim:  
        - Assign a confidence level (High / Medium / Low)  
        - Note any missing data or ambiguity  
        - Keep reasoning falsifiable and transparent  

        ---

        ### 7. **🔍 Deep Dive Investigation**
        - Select the **leading scenario** after elimination.  
        - Dive deeper:  
        - Re‑analyze supporting regions in higher detail  
        - Look for confirmatory evidence (e.g. symbols, interactions, micro‑texture)  
        - Now: **directly answer the user's question** with specific detail  

        ---

        ### 8. **Final Report**
        - Deliver structured findings:  
        - 🗺 Global context  
        - 🔍 Object‑level bullet points  
        - 🚫 Eliminated scenarios and reasons  
        - ✅ Final grounded answer: what is **known**, what is **possible**, what is **uncertain** — and **why**  

        ---

        ### 9. **Retrospective Deep Analysis**
        - After the Final Report, conduct a **retrospective reflection**:  
        - Review each phase's key insights and eliminations  
        - Highlight any lingering ambiguities or areas for further exploration  
        - Synthesize a **final, in‑depth overview** that ties together visual evidence, context, and confidence levels  
        - Use this to refine and **strengthen** the ultimate answer  

        ---

        ### 🛑 Constraints
        - **Never skip the question: `What is this image?`**  
        - **Do not form conclusions based on user language**  
        - **Always eliminate weak paths — avoid returning to them**  
        - **Always ground search using the detected language from the image itself**  
        - **Respect this full loop: observe → infer → align → analyze → eliminate → dive → synthesize → retrospective**
    """),
    stream_intermediate_steps=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    expected_output="Provide your final answer to the user."
)
from agno.playground.settings import PlaygroundSettings

settings = PlaygroundSettings(
    cors_origin_list=[
        "*",  # your frontend dev URL
    ]
)
# Ensure the folders 'saved_images' and 'processed_images' exist
for folder in ["saved_images", "processed_images"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

app = Playground(agents=[agent], settings=settings).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True, port=6789)
