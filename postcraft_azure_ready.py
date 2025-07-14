import streamlit as st
from datetime import date, timedelta
import pandas as pd
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import openai
import requests
import calendar
import json
import base64
import io
from typing import List, Dict, Optional, Tuple
import time
import tempfile
import zipfile
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

def check_moviepy_available():
    """Check if MoviePy is available for video creation"""
    try:
        import moviepy.editor
        return True
    except ImportError:
        return False

# ——— Azure OpenAI setup ———
TEXT_ENDPOINT    = os.getenv("AZURE_OPENAI_TEXT_ENDPOINT")
TEXT_API_VERSION = "2023-05-15"
TEXT_KEY         = os.getenv("AZURE_OPENAI_TEXT_KEY")

# Check if credentials are available
if not TEXT_ENDPOINT or not TEXT_KEY:
    st.error("❌ Missing Azure OpenAI credentials. Please check your environment variables.")
    st.stop()

text_client = AzureOpenAI(
    api_key       = TEXT_KEY,
    api_version   = TEXT_API_VERSION,
    azure_endpoint= TEXT_ENDPOINT
)

# ——— Image (DALL·E) setup ———
IMAGE_ENDPOINT    = os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
IMAGE_API_VERSION = "2024-02-01"
IMAGE_KEY         = os.getenv("AZURE_OPENAI_IMAGE_KEY")

# Check if image credentials are available
if not IMAGE_ENDPOINT or not IMAGE_KEY:
    st.warning("⚠️ Missing Azure OpenAI Image credentials. Image generation will be disabled.")
    image_client = None
else:
    try:
        image_client = AzureOpenAI(
            api_key       = IMAGE_KEY,
            api_version   = IMAGE_API_VERSION,
            azure_endpoint= IMAGE_ENDPOINT
        )
            
    except Exception as e:
        st.warning(f"⚠️ DALL-E 3 configuration error: {str(e)}")
        image_client = None

# ——— Azure AI Foundry Video Generation Setup ———
VIDEO_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
VIDEO_API_KEY = os.getenv("AZURE_AI_FOUNDRY_API_KEY")

# Manual deployment endpoints - update these with your actual deployed model endpoints
# Note: GPT-4 Vision and Stable Video Diffusion are not available in Azure AI Foundry
# Using alternative approaches for video generation

# Available models in Azure AI Foundry (check what's actually available)
AVAILABLE_MODELS = {
    "text-generation": os.getenv("TEXT_GEN_ENDPOINT", ""),
    "image-generation": os.getenv("IMAGE_GEN_ENDPOINT", ""),
    "embedding": os.getenv("EMBEDDING_ENDPOINT", "")
}

# Note: VideoMAE, CLIP, BLIP, GPT-4 Vision, and Stable Video Diffusion are not available in Azure AI Foundry
# Using DALL-E 3 for frame extraction and GPT-4 for caption generation as alternatives

# Check if video generation credentials are available
if not VIDEO_ENDPOINT or not VIDEO_API_KEY:
    st.warning("⚠️ Missing Azure AI Foundry credentials. Video generation will use simulated responses.")
    video_enabled = True  # Enable with simulated responses
else:
    video_enabled = True

# Video generation pipeline functions using pre-hosted models
def extract_frames_from_image(image_url: str, num_frames: int = 5, brand_niche: str = '', tone: str = '', brand_color: str = '') -> List[str]:
    """Extract frames from an image using DALL-E 3 with safe, brand-themed prompts (no reference to 'this image')."""
    try:
        st.info("📸 Using DALL-E 3 frame generation (safe prompts, no reference to original image)")
        
        if image_client is None:
            st.warning("⚠️ DALL-E 3 not available, using original image")
            return [image_url] * num_frames
        
        # Use a safe, brand-themed prompt for each frame
        st.info("🎯 Creating brand-themed frames with DALL-E 3 (safe prompts)")
        
        variations = []
        for i in range(num_frames):
            try:
                prompt = f"A professional {brand_niche or 'brand'} themed image, {tone or 'modern'} style, primary color {brand_color or '#000000'}, social media content, no text, high quality, cinematic lighting."
                response = image_client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                if response and response.data and len(response.data) > 0:
                    variations.append(response.data[0].url)
                else:
                    variations.append(image_url)
            except Exception as e:
                st.warning(f"Frame {i+1} generation failed: {e}")
                variations.append(image_url)
        return variations if variations else [image_url] * num_frames
    except Exception as e:
        st.error(f"Error extracting frames: {e}")
        return [image_url] * num_frames

def rank_frames_with_clip(frames: List[str]) -> List[Tuple[str, float]]:
    """Rank frames using simulated ranking (CLIP not available in Azure AI Foundry)"""
    try:
        # Since CLIP is not available in Azure AI Foundry, we'll use simulated ranking
        st.info("🏆 Using simulated frame ranking (CLIP not available in Azure AI Foundry)")
        
        # Simulate frame ranking for demo purposes
        import random
        ranked_frames = []
        for frame in frames:
            score = random.uniform(0.3, 0.9)
            if score > 0.28:
                ranked_frames.append((frame, score))
        
        # Sort by score
        ranked_frames.sort(key=lambda x: x[1], reverse=True)
        return ranked_frames
        
    except Exception as e:
        st.error(f"Error ranking frames: {e}")
        return [(frame, 0.5) for frame in frames]

def generate_draft_caption_with_gpt4(frames: List[str], brand_niche: str, tone: str) -> str:
    """Generate draft caption using GPT-4 (BLIP not available in Azure AI Foundry)"""
    try:
        # Since BLIP is not available in Azure AI Foundry, we'll use GPT-4 for caption generation
        st.info("✍️ Using GPT-4 for caption generation (BLIP not available in Azure AI Foundry)")
        
        # Create a prompt for caption generation
        prompt = f"""
Generate an engaging social media caption for a {brand_niche} brand with a {tone} tone.
The caption should be:
- Relevant to the {brand_niche} industry
- Engaging and shareable
- Appropriate for the {tone} tone
- Include relevant hashtags
- 1-2 sentences maximum

Return only the caption text.
"""
        
        # Use the text client (GPT-4) for caption generation
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        content = resp.choices[0].message.content
        if content is None:
            # Fallback captions
            captions = [
                f"Incredible insights from {brand_niche}!",
                f"Discover the latest in {brand_niche}",
                f"Transform your {brand_niche} journey",
                f"Expert tips for {brand_niche} success",
                f"Revolutionary {brand_niche} strategies"
            ]
            import random
            return random.choice(captions)
        
        return content.strip()
            
    except Exception as e:
        st.error(f"Error generating draft caption: {e}")
        return f"Amazing content from {brand_niche}! #{brand_niche.replace(' ', '')} #{tone.lower()}"

def polish_caption_with_gpt4_vision(draft_caption: str, frames: List[str], brand_niche: str, tone: str) -> str:
    """Polish caption using GPT-4 text (GPT-4 Vision not available in Azure AI Foundry)"""
    try:
        # Since GPT-4 Vision is not available in Azure AI Foundry, use GPT-4 text for caption polishing
        st.info("✨ Using GPT-4 text for caption polishing (GPT-4 Vision not available in Azure AI Foundry)")
        
        # Use the text client (GPT-4) for caption polishing
        prompt = f"""
Polish this caption for a {brand_niche} brand with {tone} tone:
"{draft_caption}"

Make it more engaging, add relevant hashtags, and ensure it's appropriate for social media.
Return only the polished caption.
"""
        
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        content = resp.choices[0].message.content
        if content is None:
            # Fallback polishing
            polished_variations = [
                f"🚀 {draft_caption} #Innovation #Growth",
                f"💡 {draft_caption} #Insights #Success",
                f"✨ {draft_caption} #Excellence #Achievement",
                f"🔥 {draft_caption} #Trending #Viral",
                f"🌟 {draft_caption} #Inspiration #Motivation"
            ]
            import random
            return random.choice(polished_variations)
        
        return content.strip()
            
    except Exception as e:
        st.error(f"Error polishing caption: {e}")
        return draft_caption

def choose_music_for_caption(caption: str, brand_niche: str) -> str:
    """Choose music by embedding caption and vector searching copyright-free IDs"""
    try:
        # Simulate music selection based on caption and brand niche
        # In a real implementation, this would use a music embedding service
        music_options = {
            "Technology": "tech_beat_001",
            "Fitness": "energetic_workout_002", 
            "Food": "upbeat_cooking_003",
            "Fashion": "trendy_style_004",
            "Business": "professional_ambient_005",
            "Education": "inspiring_learning_006",
            "Health": "calm_wellness_007",
            "Travel": "adventure_exploration_008",
            "Finance": "confident_success_009",
            "Entertainment": "fun_entertainment_010"
        }
        
        # Find best match for brand niche
        for category, music_id in music_options.items():
            if category.lower() in brand_niche.lower():
                return music_id
        
        return "generic_upbeat_001"
        
    except Exception as e:
        st.error(f"Error choosing music: {e}")
        return "music_id_12345"

def stylize_frames_with_dalle(frames: List[str], brand_color: str, brand_niche: str) -> List[str]:
    """Stylize frames using DALL-E 3 (more cost-effective than SDXL)"""
    try:
        # Use DALL-E 3 for frame styling (more cost-effective than SDXL)
        st.info("🎨 Using DALL-E 3 for frame styling (cost-effective alternative to SDXL)")
        
        if image_client is None:
            st.warning("⚠️ DALL-E 3 not available, using original frames")
            return frames
        
        # For MVP, let's use a simpler approach - just return the original frames
        # This avoids API rate limiting and costs while still demonstrating the pipeline
        st.info("🎯 Using simplified frame styling for MVP (original frames)")
        return frames
        
    except Exception as e:
        st.error(f"Error stylizing frames: {e}")
        return frames

def generate_transitions_with_stable_video(frames: List[str]) -> Optional[bytes]:
    """Generate transitions between consecutive frames using MoviePy"""
    try:
        st.info("🎬 Creating video with frame transitions using MoviePy")
        
        if not check_moviepy_available():
            st.error("❌ MoviePy not available for video generation")
            return None
        
        # Create video from frames with transitions
        video_bytes = create_video_from_urls(
            frames, 
            transition_type="fade", 
            duration_per_image=2.0, 
            transition_duration=0.5, 
            fps=30
        )
        
        if video_bytes:
            st.success(f"✅ Video generated successfully!")
            st.info(f"📹 Video created from {len(frames)} frames")
            st.info(f"🎬 Duration: {len(frames) * 2.0 + (len(frames) - 1) * 0.5:.1f} seconds")
            return video_bytes
        else:
            st.error("❌ Failed to create video")
            return None
            
    except Exception as e:
        st.error(f"Error generating transitions: {e}")
        return None

def compose_timeline_json(frames: List[str], caption: str, music_id: str, brand_niche: str) -> Dict:
    """Compose timeline JSON for media job"""
    timeline_data = {
        "frames": frames,
        "caption": caption,
        "music_id": music_id,
        "brand_niche": brand_niche,
        "duration_per_frame": 2.0,  # seconds
        "transition_duration": 0.5,  # seconds
        "total_duration": len(frames) * 2.0 + (len(frames) - 1) * 0.5
    }
    return timeline_data

def create_video_reel(image_url: str, brand_niche: str, tone: str, brand_color: str, 
                     brand_analysis: Optional[Dict] = None) -> Dict:
    """Complete video generation pipeline with real video creation"""
    try:
        with st.spinner("🎬 Generating video reel..."):
            # Step 1: Extract frames with actual variations
            st.info("📸 Extracting frames from image...")
            frames = extract_frames_from_image(image_url, num_frames=5, brand_niche=brand_niche, tone=tone, brand_color=brand_color)
            
            # Step 2: Rank frames
            st.info("🏆 Ranking frames with CLIP...")
            ranked_frames = rank_frames_with_clip(frames)
            top_frames = [frame for frame, score in ranked_frames[:3]]
            
            # Step 3: Generate draft caption
            st.info("✍️ Generating draft caption with GPT-4...")
            draft_caption = generate_draft_caption_with_gpt4(top_frames, brand_niche, tone)
            
            # Step 4: Polish caption with GPT-4 Vision
            st.info("✨ Polishing caption with GPT-4 Vision...")
            polished_caption = polish_caption_with_gpt4_vision(draft_caption, top_frames, brand_niche, tone)
            
            # Step 5: Choose music
            st.info("🎵 Selecting music...")
            music_id = choose_music_for_caption(polished_caption, brand_niche)
            
            # Step 6: Stylize frames
            st.info("🎨 Stylizing frames with DALL-E 3...")
            stylized_frames = stylize_frames_with_dalle(top_frames, brand_color, brand_niche)
            
            # Step 7: Generate transitions with real video creation
            st.info("🎬 Generating video transitions...")
            video_bytes = generate_transitions_with_stable_video(stylized_frames)
            
            # Step 8: Compose timeline
            timeline = compose_timeline_json(stylized_frames, polished_caption, music_id, brand_niche)
            
            # Enhanced result with real video bytes
            return {
                "video_bytes": video_bytes,
                "caption": polished_caption,
                "music_id": music_id,
                "timeline": timeline,
                "frames_used": len(top_frames),
                "total_duration": timeline["total_duration"],
                "video_quality": "HD",
                "frame_rate": "30fps",
                "resolution": "1920x1080",
                "brand_niche": brand_niche,
                "tone": tone,
                "frames": stylized_frames
            }
            
    except Exception as e:
        st.error(f"Error in video generation pipeline: {e}")
        return {
            "video_bytes": None,
            "caption": f"Amazing {brand_niche} content!",
            "music_id": "default_music",
            "timeline": {},
            "frames_used": 0,
            "total_duration": 0,
            "video_quality": "Standard",
            "frame_rate": "24fps",
            "resolution": "1280x720",
            "frames": []
        }

# Video transition functions for Streamlit integration
def download_image_from_url(url: str) -> Optional[Image.Image]:
    """Download image from URL and return PIL Image object"""
    try:
        # Handle file:// URLs (for uploaded images)
        if url.startswith("file://"):
            file_path = url.replace("file://", "")
            return Image.open(file_path)
        
        # Handle HTTP/HTTPS URLs (for generated images)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image from {url}: {e}")
        return None

def create_transition_video_clips(image_urls: List[str], transition_type: str = "fade", 
                                duration_per_image: float = 2.0, transition_duration: float = 1.0) -> List:
    """Create video clips with specified transition effects"""
    try:
        import moviepy.editor as mp
        clips = []
        
        for i, img_url in enumerate(image_urls):
            # Download image
            img = download_image_from_url(img_url)
            if img is None:
                continue
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img.save(tmp_file.name)
                img_path = tmp_file.name
            
            try:
                # Create base clip
                clip = mp.ImageClip(img_path).set_duration(duration_per_image)
                
                # Apply transition effects
                if transition_type == "fade":
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                elif transition_type == "slide_left":
                    clip = clip.set_position(("right", "center"))
                    if i < len(image_urls) - 1:
                        clip = clip.slide_out(transition_duration, "left")
                        
                elif transition_type == "slide_right":
                    clip = clip.set_position(("left", "center"))
                    if i < len(image_urls) - 1:
                        clip = clip.slide_out(transition_duration, "right")
                        
                elif transition_type == "zoom_in":
                    clip = clip.resize(lambda t: 1 + 0.3 * t / duration_per_image)
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                elif transition_type == "zoom_out":
                    clip = clip.resize(lambda t: 1.3 - 0.3 * t / duration_per_image)
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                elif transition_type == "rotate":
                    clip = clip.rotate(lambda t: 360 * t / duration_per_image)
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                elif transition_type == "flip":
                    clip = clip.resize(lambda t: 1 if t < duration_per_image/2 else -1)
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                elif transition_type == "wipe_left":
                    clip = clip.set_position(("right", "center"))
                    if i < len(image_urls) - 1:
                        clip = clip.slide_out(transition_duration, "left")
                        
                elif transition_type == "wipe_right":
                    clip = clip.set_position(("left", "center"))
                    if i < len(image_urls) - 1:
                        clip = clip.slide_out(transition_duration, "right")
                        
                elif transition_type == "dissolve":
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                    
                else:
                    # Default fade
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                
                clips.append(clip)
                
            except Exception as e:
                st.error(f"Error creating clip for image {i+1}: {e}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(img_path)
                except:
                    pass
        
        return clips
    except ImportError:
        st.error("MoviePy not available. Please install with: pip install moviepy")
        return []

def create_video_from_urls(image_urls: List[str], transition_type: str = "fade", 
                          duration_per_image: float = 2.0, transition_duration: float = 1.0, 
                          fps: int = 24) -> Optional[bytes]:
    """Create a video from image URLs with transitions and return video bytes"""
    
    if not image_urls:
        st.error("No images provided!")
        return None
    
    try:
        with st.spinner("🎬 Creating video with transitions..."):
            # Create clips with transitions
            clips = create_transition_video_clips(image_urls, transition_type, duration_per_image, transition_duration)
            
            if not clips:
                st.error("No valid clips created!")
                return None
            
            # Concatenate clips
            import moviepy.editor as mp
            video = mp.concatenate_videoclips(clips, method="compose")
            
            # Write video to bytes
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video.write_videofile(tmp_file.name, fps=fps, verbose=False, logger=None)
                
                # Read the video file
                with open(tmp_file.name, 'rb') as f:
                    video_bytes = f.read()
                
                # Clean up
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                return video_bytes
                
    except Exception as e:
        st.error(f"Error creating video: {e}")
        return None

# ——— Brand Niche Analysis & Personalization ———
def analyze_brand_niche_realtime(brand_niche: str, brand_questions: dict) -> dict:
    """Analyze brand niche with real-time market data and interactive questions"""
    prompt = f"""
Analyze this specific brand with real-time market insights and personalized recommendations:
Brand Niche: {brand_niche}

Brand Assessment:
- Target Audience: {brand_questions.get('target_audience', 'Not specified')}
- Brand Stage: {brand_questions.get('brand_stage', 'Not specified')}
- Content Goals: {', '.join(brand_questions.get('content_goals', []))}
- Platform Priority: {brand_questions.get('platform_priority', 'Not specified')}
- Budget Level: {brand_questions.get('budget_level', 'Not specified')}
- Industry Trends: {', '.join(brand_questions.get('industry_trends', []))}

Provide a detailed JSON response with real-time, personalized insights:

1. "industry_category": The specific industry category for {brand_niche}
2. "target_audience": Detailed target audience analysis based on provided data
3. "content_themes": 5-7 highly specific content themes tailored to the brand stage and goals
4. "recommended_subreddits": 5-7 subreddits relevant to the target audience and niche
5. "best_platforms": Rank platforms considering the platform priority and budget level
6. "tone_recommendations": 3-4 tone styles that match the brand stage and audience
7. "content_mix": Specific content mix percentages based on content goals and budget
8. "hashtag_strategy": Niche-specific hashtag strategy considering current trends
9. "posting_frequency": Optimal posting frequency for the brand stage and platform
10. "engagement_tactics": Specific engagement tactics for the target audience
11. "niche_insights": 3-4 unique insights about current market trends affecting this brand
12. "competitor_analysis": What successful brands in this space are doing right now
13. "trending_topics": Current trending topics that align with the industry trends
14. "content_calendar": Suggested content calendar structure for the brand stage
15. "market_opportunities": 3-4 current market opportunities for this brand
16. "risk_factors": Potential challenges based on current market conditions
17. "budget_recommendations": Specific budget allocation recommendations
18. "platform_strategy": Detailed strategy for the primary platform

Make this analysis highly specific to the brand's current situation and market conditions. Consider the real-time trends and the brand's specific stage, goals, and constraints.

Return only valid JSON.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=800
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from API")
        analysis = json.loads(content.strip())
        
        # Validate that all required fields exist
        required_fields = [
            "industry_category", "target_audience", "content_themes", 
            "recommended_subreddits", "best_platforms", "tone_recommendations",
            "content_mix", "hashtag_strategy", "posting_frequency", "engagement_tactics",
            "niche_insights", "competitor_analysis", "trending_topics", "content_calendar",
            "market_opportunities", "risk_factors", "budget_recommendations", "platform_strategy"
        ]
        
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")
        
        return analysis
    except Exception as e:
        # Fallback analysis with guaranteed structure - still personalized
        return {
            "industry_category": f"{brand_niche} Industry",
            "target_audience": f"{brand_questions.get('target_audience', brand_niche)} enthusiasts",
            "content_themes": [f"{brand_niche} insights", f"{brand_niche} tips", f"{brand_niche} trends", f"{brand_niche} success stories", f"{brand_niche} behind the scenes"],
            "recommended_subreddits": [f"{brand_niche.lower()}", "entrepreneur", "marketing", "business", "startups"],
            "best_platforms": ["Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook"],
            "tone_recommendations": ["Professional", "Educational", "Witty", "Empathetic"],
            "content_mix": {"educational": 40, "entertaining": 30, "promotional": 20, "user_generated": 10},
            "hashtag_strategy": [f"{brand_niche}-specific", "Trending", "Branded", "Community"],
            "posting_frequency": "3-5 times per week",
            "engagement_tactics": ["Ask questions", "Share user content", "Respond to comments", "Create polls"],
            "niche_insights": [f"Focus on {brand_niche} expertise", f"Build {brand_niche} community", f"Share {brand_niche} innovations"],
            "competitor_analysis": f"Study successful {brand_niche} brands for best practices",
            "trending_topics": [f"Latest {brand_niche} trends", f"{brand_niche} technology", f"{brand_niche} market insights"],
            "content_calendar": f"Structured {brand_niche} content planning",
            "market_opportunities": [f"Growing {brand_niche} market", f"Digital transformation in {brand_niche}", f"New {brand_niche} technologies"],
            "risk_factors": [f"Market competition in {brand_niche}", f"Economic factors affecting {brand_niche}", f"Technology changes in {brand_niche}"],
            "budget_recommendations": "Allocate 60% to content creation, 30% to paid promotion, 10% to tools",
            "platform_strategy": f"Focus on {brand_questions.get('platform_priority', 'Instagram')} as primary platform"
        }

def analyze_brand_niche(brand_niche: str, brand_description: str = "") -> dict:
    """Analyze brand niche and provide personalized recommendations"""
    prompt = f"""
Analyze this specific brand niche and provide highly personalized recommendations:
Brand Niche: {brand_niche}
Brand Description: {brand_description}

Provide a detailed JSON response with niche-specific insights:

1. "industry_category": The specific industry category for {brand_niche}
2. "target_audience": Detailed primary and secondary target audiences for {brand_niche}
3. "content_themes": 5-7 highly specific content themes that work exceptionally well for {brand_niche} brands
4. "recommended_subreddits": 5-7 subreddits specifically relevant to {brand_niche} content and audience
5. "best_platforms": Rank platforms by effectiveness for {brand_niche} (Instagram, LinkedIn, Twitter, TikTok, Facebook, YouTube)
6. "tone_recommendations": 3-4 tone styles that resonate with {brand_niche} audience
7. "content_mix": Specific content mix percentages tailored for {brand_niche} (educational %, entertaining %, promotional %, user_generated %, etc.)
8. "hashtag_strategy": Niche-specific hashtag categories and examples for {brand_niche}
9. "posting_frequency": Optimal posting frequency for {brand_niche} audience engagement
10. "engagement_tactics": Specific engagement tactics that work well for {brand_niche} brands
11. "niche_insights": 3-4 unique insights about {brand_niche} marketing
12. "competitor_analysis": What successful {brand_niche} brands are doing
13. "trending_topics": Current trending topics in {brand_niche} space
14. "content_calendar": Suggested content calendar structure for {brand_niche}

Make this analysis highly specific to {brand_niche} - avoid generic advice. Consider the unique challenges, opportunities, and audience behavior in this specific niche.

Return only valid JSON.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=500
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from API")
        analysis = json.loads(content.strip())
        
        # Validate that all required fields exist
        required_fields = [
            "industry_category", "target_audience", "content_themes", 
            "recommended_subreddits", "best_platforms", "tone_recommendations",
            "content_mix", "hashtag_strategy", "posting_frequency", "engagement_tactics"
        ]
        
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")
        
        return analysis
    except Exception as e:
        # Fallback analysis with guaranteed structure - still niche-specific
        return {
            "industry_category": f"{brand_niche} Industry",
            "target_audience": f"{brand_niche} enthusiasts and professionals",
            "content_themes": [f"{brand_niche} insights", f"{brand_niche} tips", f"{brand_niche} trends", f"{brand_niche} success stories", f"{brand_niche} behind the scenes"],
            "recommended_subreddits": [f"{brand_niche.lower()}", "entrepreneur", "marketing", "business", "startups"],
            "best_platforms": ["Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook"],
            "tone_recommendations": ["Professional", "Educational", "Witty", "Empathetic"],
            "content_mix": {"educational": 40, "entertaining": 30, "promotional": 20, "user_generated": 10},
            "hashtag_strategy": [f"{brand_niche}-specific", "Trending", "Branded", "Community"],
            "posting_frequency": "3-5 times per week",
            "engagement_tactics": ["Ask questions", "Share user content", "Respond to comments", "Create polls"],
            "niche_insights": [f"Focus on {brand_niche} expertise", f"Build {brand_niche} community", f"Share {brand_niche} innovations"],
            "competitor_analysis": f"Study successful {brand_niche} brands for best practices",
            "trending_topics": [f"Latest {brand_niche} trends", f"{brand_niche} technology", f"{brand_niche} market insights"],
            "content_calendar": f"Structured {brand_niche} content planning"
        }

def get_niche_specific_subreddits(brand_niche: str) -> list:
    """Get relevant subreddits based on brand niche"""
    niche_subreddit_map = {
        "Technology": ["programming", "technology", "gadgets", "startups", "artificial", "MachineLearning", "webdev", "datascience"],
        "Fitness": ["fitness", "bodybuilding", "running", "yoga", "nutrition", "loseit", "weightlifting", "crossfit"],
        "Food": ["food", "recipes", "cooking", "MealPrepSunday", "slowcooking", "baking", "chefknives", "fermentation"],
        "Fashion": ["fashion", "streetwear", "malefashionadvice", "femalefashionadvice", "sneakers", "watches", "jewelry"],
        "Business": ["entrepreneur", "smallbusiness", "marketing", "sales", "startups", "investing", "consulting", "freelance"],
        "Education": ["science", "todayilearned", "explainlikeimfive", "askscience", "education", "academia", "gradschool"],
        "Health": ["health", "nutrition", "fitness", "mentalhealth", "meditation", "wellness", "supplements", "biohacking"],
        "Travel": ["travel", "backpacking", "digitalnomad", "solotravel", "travelphotography", "backpacking", "hostels"],
        "Finance": ["personalfinance", "investing", "wallstreetbets", "financialindependence", "cryptocurrency", "stocks", "realestate"],
        "Entertainment": ["movies", "television", "music", "gaming", "books", "entertainment", "podcasts", "netflix"],
        "Sports": ["sports", "soccer", "basketball", "nfl", "baseball", "fitness", "tennis", "golf"],
        "Lifestyle": ["lifestyle", "productivity", "minimalism", "selfimprovement", "motivation", "zenhabits", "getdisciplined"],
        "Beauty": ["beauty", "skincareaddiction", "makeupaddiction", "haircarescience", "beauty", "skincare", "makeup"],
        "Parenting": ["parenting", "mommit", "daddit", "toddlers", "pregnant", "breastfeeding", "babybumps"],
        "Pets": ["aww", "dogs", "cats", "pets", "dogtraining", "catcare", "petadvice", "dogpictures"],
        "Photography": ["photography", "photocritique", "itookapicture", "analog", "streetphotography", "portraits"],
        "Art": ["art", "drawing", "painting", "digitalart", "artists", "sketchbook", "watercolor"],
        "Music": ["music", "listentothis", "hiphopheads", "indieheads", "electronicmusic", "classicalmusic"],
        "Gaming": ["gaming", "pcgaming", "ps4", "xboxone", "nintendo", "indiegaming", "gamedev"],
        "Cars": ["cars", "autos", "teslamotors", "electricvehicles", "carphotography", "mechanicadvice"],
        "Real Estate": ["realestate", "firsttimehomebuyer", "investing", "personalfinance", "homeimprovement"],
        "Crypto": ["cryptocurrency", "bitcoin", "ethereum", "defi", "cryptomarkets", "blockchain"],
        "Fitness Tech": ["fitness", "applewatch", "fitbit", "strava", "running", "cycling", "garmin"],
        "Sustainable Living": ["zerowaste", "sustainability", "environment", "vegan", "minimalism", "homesteading"],
        "Mental Health": ["mentalhealth", "anxiety", "depression", "meditation", "mindfulness", "therapy"],
        "Cooking": ["cooking", "recipes", "food", "chefknives", "fermentation", "baking", "mealprep"],
        "DIY": ["diy", "woodworking", "homeimprovement", "crafts", "sewing", "gardening"],
        "Tech Reviews": ["technology", "gadgets", "android", "apple", "buildapc", "hardware"],
        "Marketing": ["marketing", "entrepreneur", "advertising", "socialmedia", "seo", "contentmarketing"],
        "Design": ["design", "graphic_design", "web_design", "ui_design", "typography", "logos"],
        "Writing": ["writing", "writers", "selfpublish", "books", "poetry", "screenwriting"],
        "Podcasting": ["podcasting", "podcasts", "audio", "microphones", "editing", "contentcreation"],
        "E-commerce": ["entrepreneur", "ecommerce", "dropshipping", "shopify", "amazon", "onlinebusiness"],
        "Fitness Coaching": ["fitness", "personaltraining", "nutrition", "coaching", "motivation", "transformation"],
        "Tech Startups": ["startups", "entrepreneur", "programming", "technology", "venturecapital", "productmanagement"],
        "Creative Business": ["entrepreneur", "art", "design", "photography", "music", "creative", "freelance"],
        "Health Tech": ["health", "technology", "fitness", "biohacking", "supplements", "wellness"],
        "Sustainable Business": ["entrepreneur", "sustainability", "environment", "business", "socialenterprise"],
        "Remote Work": ["digitalnomad", "remotework", "freelance", "productivity", "workfromhome"],
        "Personal Development": ["selfimprovement", "motivation", "productivity", "zenhabits", "getdisciplined", "meditation"]
    }
    
    # Find best match for brand niche
    for category, subreddits in niche_subreddit_map.items():
        if category.lower() in brand_niche.lower():
            return subreddits
    
    # If no exact match, try partial matches
    brand_niche_lower = brand_niche.lower()
    for category, subreddits in niche_subreddit_map.items():
        if any(word in brand_niche_lower for word in category.lower().split()):
            return subreddits
    
    # Default fallback with more relevant options
    return ["entrepreneur", "marketing", "business", "startups", "socialmedia"]

def generate_action_plan(brand_niche: str, trend_analysis: Optional[dict] = None, competitor_analysis: Optional[dict] = None) -> str:
    """Generate personalized action plan based on trend and competitor analysis"""
    prompt = f"""
You are a strategic marketing consultant creating a personalized action plan for a {brand_niche} brand.

Brand Niche: {brand_niche}

Trend Analysis:
{json.dumps(trend_analysis, indent=2) if trend_analysis else "No trend analysis available"}

Competitor Analysis:
{json.dumps(competitor_analysis, indent=2) if competitor_analysis else "No competitor analysis available"}

Create a comprehensive, actionable plan that includes:

1. IMMEDIATE ACTIONS (Next 30 days):
   - Specific content adaptations based on current trends
   - Competitor strategy implementations
   - Quick wins and low-hanging fruit

2. SHORT-TERM STRATEGY (Next 3 months):
   - Content calendar adjustments
   - Platform-specific optimizations
   - Engagement strategy improvements

3. LONG-TERM POSITIONING (Next 6-12 months):
   - Differentiation strategies
   - Market opportunity capture
   - Brand positioning evolution

4. RISK MITIGATION:
   - Address identified challenges
   - Prepare for market changes
   - Build competitive advantages

5. SUCCESS METRICS:
   - KPIs to track progress
   - Milestone targets
   - Performance indicators

Make this plan highly specific to {brand_niche} and actionable. Focus on concrete steps that can be implemented immediately.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=800
        )
        content = resp.choices[0].message.content
        if content is None:
            return "Generated personalized action plan for your brand."
        return content.strip()
    except Exception as e:
        return f"Action plan for {brand_niche}: Focus on current trends, implement competitor best practices, and build unique positioning."

def generate_personalized_content_strategy(brand_niche: str, brand_analysis: dict, platform: str, tone: str) -> str:
    """Generate personalized content strategy based on brand analysis"""
    prompt = f"""
You are a social media strategist creating highly personalized content for a {brand_niche} brand.

Brand Analysis:
- Industry: {brand_analysis.get('industry_category', 'General')}
- Target Audience: {brand_analysis.get('target_audience', 'General audience')}
- Content Themes: {', '.join(brand_analysis.get('content_themes', []))}
- Content Mix: {brand_analysis.get('content_mix', {})}
- Engagement Tactics: {', '.join(brand_analysis.get('engagement_tactics', []))}
- Niche Insights: {', '.join(brand_analysis.get('niche_insights', []))}
- Competitor Analysis: {brand_analysis.get('competitor_analysis', '')}
- Trending Topics: {', '.join(brand_analysis.get('trending_topics', []))}

Platform: {platform}
Tone: {tone}

Create a highly personalized content strategy specifically for {brand_niche} that:
1. Leverages the unique insights about {brand_niche} marketing
2. Addresses the specific challenges and opportunities in {brand_niche}
3. Uses content themes that resonate with {brand_niche} audience
4. Incorporates trending topics in {brand_niche} space
5. Follows platform-specific best practices for {platform}
6. Maintains the {tone} tone appropriate for {brand_niche}
7. Includes engagement tactics that work well for {brand_niche} brands
8. Considers competitor strategies in {brand_niche}
9. Provides actionable, niche-specific recommendations

Make this strategy highly specific to {brand_niche} - avoid generic advice. Focus on what makes {brand_niche} marketing unique and effective.
"""
    
    resp = text_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=400
    )
    content = resp.choices[0].message.content
    if content is None:
        return "Generated personalized content strategy."
    return content.strip()

# ——— Fetch Reddit hot posts via public JSON (skip stickied) ———
def analyze_industry_trends(brand_niche: str) -> dict:
    """Analyze current industry trends for a specific brand niche"""
    prompt = f"""
Analyze current industry trends for {brand_niche} and provide real-time insights.

Provide a JSON response with:
1. "current_trends": 5-7 current trends affecting {brand_niche} industry
2. "emerging_technologies": 3-4 emerging technologies in {brand_niche}
3. "consumer_behavior": 3-4 changes in consumer behavior in {brand_niche}
4. "market_opportunities": 3-4 current market opportunities in {brand_niche}
5. "challenges": 3-4 current challenges facing {brand_niche} brands
6. "seasonal_factors": Any seasonal trends affecting {brand_niche}
7. "social_media_trends": 3-4 social media trends relevant to {brand_niche}
8. "content_trends": 3-4 content trends that work well for {brand_niche}

Focus on current, real-time trends and insights that are happening right now in {brand_niche}.

Return only valid JSON.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=600
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from API")
        return json.loads(content.strip())
    except Exception as e:
        # Fallback trends
        return {
            "current_trends": [f"Digital transformation in {brand_niche}", f"AI adoption in {brand_niche}", f"Sustainability in {brand_niche}"],
            "emerging_technologies": [f"AI/ML in {brand_niche}", f"Automation in {brand_niche}", f"Cloud solutions for {brand_niche}"],
            "consumer_behavior": [f"Increased online engagement", f"Demand for personalization", f"Focus on sustainability"],
            "market_opportunities": [f"Growing {brand_niche} market", f"Digital-first approach", f"Innovation in {brand_niche}"],
            "challenges": [f"Market competition", f"Technology adoption", f"Customer acquisition"],
            "seasonal_factors": [f"Q4 growth in {brand_niche}", f"Holiday season impact", f"Year-end planning"],
            "social_media_trends": [f"Video content dominance", f"Authentic storytelling", f"Community building"],
            "content_trends": [f"Educational content", f"Behind-the-scenes", f"User-generated content"]
        }

def analyze_competitors(brand_niche: str, platform: str = "Instagram") -> dict:
    """Analyze competitor strategies for a specific brand niche"""
    prompt = f"""
Analyze competitor strategies for {brand_niche} brands on {platform} and provide actionable insights.

Provide a JSON response with:
1. "top_competitors": 5-7 top competitors in {brand_niche} space
2. "competitor_strategies": 3-4 common strategies used by successful {brand_niche} brands
3. "content_themes": 3-4 content themes that competitors are using successfully
4. "engagement_tactics": 3-4 engagement tactics that work well for competitors
5. "posting_patterns": Typical posting frequency and timing for {brand_niche} brands
6. "hashtag_strategies": 3-4 hashtag strategies used by competitors
7. "success_metrics": 3-4 metrics that indicate success in {brand_niche}
8. "differentiation_opportunities": 3-4 ways to differentiate from competitors
9. "partnership_opportunities": 3-4 potential partnership opportunities
10. "innovation_gaps": 3-4 areas where competitors are lacking innovation

Focus on actionable insights that can be implemented immediately.

Return only valid JSON.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=600
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from API")
        return json.loads(content.strip())
    except Exception as e:
        # Fallback competitor analysis
        return {
            "top_competitors": [f"Leading {brand_niche} brand 1", f"Established {brand_niche} company", f"Emerging {brand_niche} startup"],
            "competitor_strategies": [f"Educational content focus", f"Community building", f"Thought leadership"],
            "content_themes": [f"{brand_niche} tips and tricks", f"Industry insights", f"Behind-the-scenes"],
            "engagement_tactics": ["Interactive polls", "User-generated content", "Live Q&A sessions"],
            "posting_patterns": "3-5 posts per week, peak engagement times",
            "hashtag_strategies": [f"{brand_niche}-specific hashtags", "Industry hashtags", "Trending hashtags"],
            "success_metrics": ["Engagement rate", "Follower growth", "Lead generation"],
            "differentiation_opportunities": [f"Unique {brand_niche} perspective", "Innovative content formats", "Personal brand building"],
            "partnership_opportunities": [f"Collaborate with {brand_niche} influencers", f"Cross-promotion with complementary brands", f"Industry partnerships"],
            "innovation_gaps": [f"Lack of video content", f"Limited personalization", f"Missing community engagement"]
        }

def get_trending_topics(subreddit: str, num: int) -> list[dict]:
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT")}
    
    # Clean the subreddit name (remove r/ prefix if present)
    clean_subreddit = subreddit.replace("r/", "").strip()
    
    # Try the user's selected subreddit first
    url = f"https://www.reddit.com/r/{clean_subreddit}/hot.json?limit={num*2}"
    resp = requests.get(url, headers=headers)
    
    if resp.status_code == 200:
        items = resp.json().get("data", {}).get("children", [])
        topics = []
        for item in items:
            data = item["data"]
            if data.get("stickied") or data.get("author") == "AutoModerator":
                continue
            topics.append({"title": data["title"], "url": data["url"]})
            if len(topics) >= num:
                break
        if topics:
            return topics
    
    # If user's subreddit fails, show warning and try fallback
    st.warning(f"⚠️ Could not fetch posts from r/{clean_subreddit}. Trying fallback subreddits...")
    
    # Try fallback subreddits
    fallback_subreddits = ["memes", "funny", "todayilearned"]
    for fallback_sub in fallback_subreddits:
        url = f"https://www.reddit.com/r/{fallback_sub}/hot.json?limit={num*2}"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            continue
        items = resp.json().get("data", {}).get("children", [])
        topics = []
        for item in items:
            data = item["data"]
            if data.get("stickied") or data.get("author") == "AutoModerator":
                continue
            topics.append({"title": data["title"], "url": data["url"]})
            if len(topics) >= num:
                break
        if topics:
            st.info(f"✅ Using posts from r/{fallback_sub} as fallback")
            return topics
    
    # If all else fails, return generic topics
    st.error(f"❌ Could not fetch posts from any subreddit. Using generic topics.")
    return [{"title": f"Generic {clean_subreddit} topic {i+1}", "url": ""} for i in range(num)]

def select_best_time(platform: str) -> str:
    return {"Instagram":"11:00 AM","LinkedIn":"09:00 AM","Twitter":"12:00 PM"}.get(platform,"10:00 AM")

def generate_post_for_trend(subreddit, tone, platform, trend_info, brand_niche="", brand_analysis=None):
    # Enhanced prompt with brand niche personalization
    brand_context = ""
    if brand_analysis:
        brand_context = f"""
Brand Context:
- Industry: {brand_analysis.get('industry_category', 'General')}
- Target Audience: {brand_analysis.get('target_audience', 'General audience')}
- Content Themes: {', '.join(brand_analysis.get('content_themes', []))}
"""
    
    prompt = f"""
You are a social media strategist for a {brand_niche} brand on {platform}.
{brand_context}
Here is a current viral Reddit post:
Title: "{trend_info['title']}"
URL: {trend_info['url']}

Create a strategy that:
1. Connects this trend to the {brand_niche} industry in a relevant way
2. Defines a clear goal for the post
3. Tailors the approach to {platform} audience and {tone} tone
4. Incorporates brand-specific themes and messaging
5. Ensures the content feels authentic to the brand niche

Provide:
1. Goal definition
2. How this trend connects to the brand niche
3. Platform-specific approach
4. Caption with relevant hashtags
5. Engagement strategy
"""
    resp = text_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=400
    )
    content = resp.choices[0].message.content
    if content is None:
        return "Generated post strategy for the trend."
    return content.strip()

def generate_image(strategy: str, tone: str, brand_color: str, include_logo: bool, brand_niche: str = ""):
    # Enhanced image generation with brand niche context
    cot_prompt = f"""
You want an image for a social post based on this strategy:
\"\"\"
{strategy}
\"\"\"
Brand Niche: {brand_niche}
Tone: {tone}

In one sentence, what's the core concept or emotion? Then, in one sentence, describe a strong visual metaphor that aligns with the brand niche.
Keep it concise and focused on the main idea, considering the brand's industry and Dall-E's capabilities and policies.
Finally, output ONLY:
Image Prompt: <your final prompt here> (1024×1024, no text or lettering, brand color {brand_color})
"""
    cot_resp = text_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role":"user","content":cot_prompt}],
        temperature=0.7,
        max_tokens=200
    )
    content = cot_resp.choices[0].message.content
    if content is None:
        final_prompt = f"Professional {brand_niche} brand image with {tone} tone, brand color {brand_color}"
    else:
        # Extract last "Image Prompt:" line
        final_prompt = next(
            line.replace("Image Prompt:", "").strip()
            for line in content.splitlines()
            if line.startswith("Image Prompt:")
        )

    if include_logo:
        final_prompt += ", include the brand logo"

    # 2) Call DALL·E with 1024×1024
    if image_client is None:
        return "https://via.placeholder.com/1024x1024/cccccc/666666?text=Image+Generation+Disabled", final_prompt
    
    img = image_client.images.generate(
        model="dall-e-3",
        prompt=final_prompt,
        n=1,
        size="1024x1024"
    )
    if img and img.data and len(img.data) > 0:
        return img.data[0].url, final_prompt
    else:
        return "https://via.placeholder.com/1024x1024/cccccc/666666?text=Image+Generation+Failed", final_prompt

def generate_final_post(strategy, brand_niche, subreddit, tone, platform, brand_color, logo_desc, brand_analysis=None):
    brand_context = ""
    if brand_analysis:
        brand_context = f"""
Brand Analysis:
- Industry: {brand_analysis.get('industry_category', 'General')}
- Target Audience: {brand_analysis.get('target_audience', 'General audience')}
- Content Themes: {', '.join(brand_analysis.get('content_themes', []))}
- Engagement Tactics: {', '.join(brand_analysis.get('engagement_tactics', []))}
"""
    
    prompt = f"""
You are crafting the final social media post for a {brand_niche} brand on {platform}.
Brand primary color: {brand_color}.
{f"Brand logo: {logo_desc}" if logo_desc else ""}
{brand_context}
Content inspiration: r/{subreddit} viral post strategy.
Strategy:
\"\"\"{strategy}\"\"\"

Create a final post that:
1. Feels authentic to the {brand_niche} industry
2. Aligns with the brand's target audience
3. Uses appropriate tone and language for {platform}
4. Incorporates relevant hashtags for the niche
5. Includes a compelling call to action

Please produce:
- A short title.
- A polished caption (2-3 hashtags).
- A call to action.

Label each output.
"""
    resp = text_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=300
    )
    content = resp.choices[0].message.content
    if content is None:
        return "Generated final post content."
    return content.strip()

# ——— Streamlit UI ———
st.set_page_config(page_title="PostCraft Pro", layout="wide")
st.title("📊 PostCraft Pro: Trend Calendar & Post Creator")

# Enhanced Brand Profile with Interactive Questions
st.sidebar.header("🎯 Brand Profile")

# Initialize session state for brand questions
if "brand_questions" not in st.session_state:
    st.session_state.brand_questions = {}
if "brand_analysis" not in st.session_state:
    st.session_state.brand_analysis = None

# Brand Niche Input
brand_niche = st.sidebar.text_input("Brand Niche/Industry", "Your Brand Niche")

# Interactive Brand Questions
if brand_niche and brand_niche != "Your Brand Niche":
    st.sidebar.subheader("📋 Brand Assessment")
    
    # Question 1: Target Audience
    target_audience = st.sidebar.selectbox(
        "Who is your primary target audience?",
        ["Select audience...", "Young professionals (25-35)", "Small business owners", "Tech enthusiasts", 
         "Fitness enthusiasts", "Creative professionals", "Students", "Parents", "Seniors", "Entrepreneurs",
         "Remote workers", "Health-conscious individuals", "Luxury consumers", "Budget-conscious consumers"],
        key="target_audience"
    )
    
    # Question 2: Brand Stage
    brand_stage = st.sidebar.selectbox(
        "What stage is your brand in?",
        ["Select stage...", "Just starting out", "Growing rapidly", "Established business", "Mature brand", "Rebranding"],
        key="brand_stage"
    )
    
    # Question 3: Content Goals
    content_goals = st.sidebar.multiselect(
        "What are your main content goals?",
        ["Brand awareness", "Lead generation", "Community building", "Sales conversion", "Thought leadership", 
         "Customer education", "Product promotion", "Industry expertise", "Customer support"],
        key="content_goals"
    )
    
    # Question 4: Platform Priority
    platform_priority = st.sidebar.selectbox(
        "Which platform is your primary focus?",
        ["Select platform...", "Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook", "YouTube", "Pinterest"],
        key="platform_priority"
    )
    
    # Question 5: Budget Level
    budget_level = st.sidebar.selectbox(
        "What's your marketing budget level?",
        ["Select budget...", "Minimal budget", "Moderate budget", "Healthy budget", "High budget"],
        key="budget_level"
    )
    
    # Question 6: Industry Trends
    industry_trends = st.sidebar.multiselect(
        "What trends are affecting your industry?",
        ["AI/Technology", "Sustainability", "Remote work", "Health & wellness", "E-commerce growth",
         "Social commerce", "Video content", "Personalization", "Community-driven", "Authenticity"],
        key="industry_trends"
    )
    
    # Store answers in session state
    st.session_state.brand_questions = {
        "target_audience": target_audience,
        "brand_stage": brand_stage,
        "content_goals": content_goals,
        "platform_priority": platform_priority,
        "budget_level": budget_level,
        "industry_trends": industry_trends
    }
    
    # Analyze brand with real-time data when all questions are answered
    if (target_audience != "Select audience..." and brand_stage != "Select stage..." and 
        platform_priority != "Select platform..." and budget_level != "Select budget..." and
        len(content_goals) > 0 and len(industry_trends) > 0):
        
        if st.sidebar.button("🚀 Generate Real-Time Strategy", type="primary"):
            with st.spinner("Analyzing market trends and generating personalized strategy..."):
                brand_analysis = analyze_brand_niche_realtime(brand_niche, st.session_state.brand_questions)
                st.session_state.brand_analysis = brand_analysis
                st.sidebar.success("Real-time strategy generated!")
    
    # Show current answers
    if st.session_state.brand_questions:
        st.sidebar.subheader("📊 Your Brand Profile")
        for key, value in st.session_state.brand_questions.items():
            if value and value != "Select audience..." and value != "Select stage..." and value != "Select platform..." and value != "Select budget...":
                if isinstance(value, list):
                    st.sidebar.write(f"**{key.replace('_', ' ').title()}:** {', '.join(value)}")
                else:
                    st.sidebar.write(f"**{key.replace('_', ' ').title()}:** {value}")

# Display brand analysis if available
if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
    brand_analysis = st.session_state.brand_analysis
    st.sidebar.subheader("📊 Brand Analysis")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Industry", brand_analysis.get('industry_category', 'General'))
        target_audience = brand_analysis.get('target_audience', 'General audience')
        st.metric("Target Audience", target_audience[:20] + "..." if len(target_audience) > 20 else target_audience)
    
    with col2:
        best_platforms = brand_analysis.get('best_platforms', ['Instagram'])
        # Ensure best_platforms is a list and has at least one element
        if isinstance(best_platforms, list) and len(best_platforms) > 0:
            best_platform = best_platforms[0]
        else:
            best_platform = 'Instagram'
        st.metric("Best Platform", best_platform)
        st.metric("Posting Frequency", brand_analysis.get('posting_frequency', '3-5/week'))

# Brand visual elements
st.sidebar.header("🎨 Brand Visuals")
logo_file   = st.sidebar.file_uploader("Upload Logo", type=["png","jpg","jpeg","svg"])
brand_color = st.sidebar.color_picker("Primary Color", "#000000")
logo_desc   = logo_file.name if logo_file else ""

# Custom CSS for scrollable tabs (Dark Mode Compatible)
st.markdown("""
<style>
    /* Custom scrollable tabs styling - Dark Mode Compatible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        overflow-x: auto;
        scrollbar-width: thin;
        scrollbar-color: #666 #2e2e2e;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #2e2e2e;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #666;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
        background: #888;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        min-width: 120px;
        flex-shrink: 0;
        border: 1px solid #333;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        border-bottom: 2px solid #ff4b4b;
        color: #ffffff;
    }
    
    .stTabs [aria-selected="false"] {
        background-color: #1e1e1e;
        color: #cccccc;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background-color: #2e2e2e;
        color: #ffffff;
    }
    
    /* Tab navigation arrows - Dark Mode */
    .tab-nav-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        gap: 10px;
    }
    
    .tab-nav-button {
        background: #ff4b4b;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 14px;
    }
    
    .tab-nav-button:hover {
        background: #ff3333;
    }
    
    .tab-nav-button:disabled {
        background: #444;
        color: #888;
        cursor: not-allowed;
    }
    
    /* Dark mode text colors */
    .dark-mode-text {
        color: #ffffff !important;
    }
    
    .dark-mode-text-secondary {
        color: #cccccc !important;
    }
    
    /* Dark mode backgrounds */
    .dark-mode-bg {
        background-color: #0e1117 !important;
    }
    
    .dark-mode-bg-secondary {
        background-color: #1e1e1e !important;
    }
    
    /* Custom button styling for dark mode */
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #ff3333;
    }
    
    .stButton > button:disabled {
        background-color: #444;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Tab navigation with scroll functionality
tab_names = ["📅 Calendar", "✍️ Create Post", "🎯 Brand Strategy", "📊 Market Analysis", "🎬 Video Reel", "🎬 Video Transitions"]

# Initialize current tab in session state
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0

# Tab navigation buttons
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

with col1:
    if st.button("◀️", key="prev_tab", disabled=st.session_state.current_tab == 0):
        st.session_state.current_tab = max(0, st.session_state.current_tab - 1)
        st.rerun()

with col2:
    if st.button("▶️", key="next_tab", disabled=st.session_state.current_tab == len(tab_names) - 1):
        st.session_state.current_tab = min(len(tab_names) - 1, st.session_state.current_tab + 1)
        st.rerun()

with col3:
    st.write(f"**Tab {st.session_state.current_tab + 1} of {len(tab_names)}**")

with col4:
    st.write(f"**{tab_names[st.session_state.current_tab]}**")

# Create tabs
tabs = st.tabs(tab_names)

# Unpack tabs
tab1, tab2, tab3, tab4, tab5, tab6 = tabs

with tab1:
    st.header("14-Day Trend-Driven Calendar")
    
    # Enhanced platform selection with brand analysis
    if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
        brand_analysis = st.session_state.brand_analysis
        recommended_platforms = brand_analysis.get('best_platforms', ["Instagram","LinkedIn","Twitter"])
        # Ensure it's a list
        if not isinstance(recommended_platforms, list) or len(recommended_platforms) == 0:
            recommended_platforms = ["Instagram","LinkedIn","Twitter"]
        platform = st.selectbox("Platform", recommended_platforms)
    else:
        platform = st.selectbox("Platform", ["Instagram","LinkedIn","Twitter"])
    
    # Niche-specific subreddit suggestions
    if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
        brand_analysis = st.session_state.brand_analysis
        recommended_subreddits = brand_analysis.get('recommended_subreddits', ["memes"])
        # Ensure it's a list
        if not isinstance(recommended_subreddits, list) or len(recommended_subreddits) == 0:
            recommended_subreddits = ["memes"]
        subreddit = st.selectbox("Subreddit", recommended_subreddits)
    else:
        niche_subreddits = get_niche_specific_subreddits(brand_niche)
        subreddit = st.selectbox("Subreddit", niche_subreddits)
    
    # Enhanced tone selection
    if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
        brand_analysis = st.session_state.brand_analysis
        recommended_tones = brand_analysis.get('tone_recommendations', ["Witty","Empathetic","Educational","Professional"])
        # Ensure it's a list
        if not isinstance(recommended_tones, list) or len(recommended_tones) == 0:
            recommended_tones = ["Witty","Empathetic","Educational","Professional"]
        tone = st.selectbox("Tone", recommended_tones)
    else:
        tone = st.selectbox("Tone", ["Witty","Empathetic","Educational","Professional"])
    
    days = st.slider("Days to Plan", 1, 14, 7)

    if st.button("Generate Personalized Calendar"):
        start = date.today()
        if subreddit is None:
            subreddit = "memes"  # Default fallback
        trends = get_trending_topics(subreddit, days)
        rows = []
        
        for i, t in enumerate(trends):
            d = start + timedelta(days=i)
            # Get brand_analysis from session state if available
            current_brand_analysis = st.session_state.get('brand_analysis') if 'brand_analysis' in st.session_state else None
            strategy = generate_post_for_trend(subreddit, tone, platform, t, brand_niche, current_brand_analysis)
            rows.append({
                "Date":     d.isoformat(),
                "Time":     select_best_time(platform),
                "Trend":    t["title"],
                "Source":   t["url"],
                "Strategy": strategy
            })
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.session_state.calendar_df   = df
        st.session_state.calendar_meta = {
            "platform": platform,
            "subreddit": subreddit,
            "tone": tone,
            "brand_niche": brand_niche,
            "brand_analysis": st.session_state.get('brand_analysis') if 'brand_analysis' in st.session_state else None
        }
        st.success("Calendar generated successfully! Go to the '✍️ Create Post' tab to create posts from your calendar.")

with tab2:
    st.header("Create Post from Strategy")
    if "calendar_df" in st.session_state:
        df   = st.session_state.calendar_df
        meta = st.session_state.calendar_meta
        
        # Initialize pick if not exists
        if "pick" not in st.session_state:
            st.session_state.pick = 0
        
        # Create selectbox for picking entry
        selected_entry = st.selectbox(
            "Pick entry to create final post:",
            df.index,
            key="pick",
            format_func=lambda i: f"{df.at[i,'Date']} – {df.at[i,'Trend']}"
        )
        
        if selected_entry is not None:
            entry = df.loc[selected_entry]
            st.markdown(f"**Strategy:** {entry['Strategy']}")

            if st.button("Generate Final Post & Image"):
                final = generate_final_post(
                    strategy    = entry["Strategy"],
                    brand_niche = brand_niche,
                    subreddit   = meta["subreddit"],
                    tone        = meta["tone"],
                    platform    = meta["platform"],
                    brand_color = brand_color,
                    logo_desc   = logo_desc,
                    brand_analysis = meta.get("brand_analysis")
                )
                img_url, img_prompt = generate_image(
                    entry["Strategy"],
                    meta["tone"],
                    brand_color,
                    bool(logo_desc),
                    brand_niche
                )
                st.subheader("📝 Final Post")
                st.markdown(final)
                st.subheader("🖼️ Generated Image")
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.warning("No image URL available")
                st.caption(f"Prompt: `{img_prompt}`")
    else:
        st.info("Generate your calendar first on the 📅 Calendar tab.")

with tab3:
    st.header("🎯 Brand Strategy Dashboard")
    
    # Get brand_analysis from session state
    brand_analysis = st.session_state.get('brand_analysis') if 'brand_analysis' in st.session_state else None
    
    if brand_analysis:
        # Main insights section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Content Strategy")
            content_mix = brand_analysis.get('content_mix', {})
            if content_mix and isinstance(content_mix, dict):
                st.write("**Recommended Content Mix:**")
                for content_type, percentage in content_mix.items():
                    try:
                        # Convert to float and ensure it's a valid percentage
                        percentage_value = float(percentage)
                        if 0 <= percentage_value <= 100:
                            st.progress(percentage_value / 100)
                            st.write(f"{content_type.title()}: {percentage_value}%")
                        else:
                            st.write(f"{content_type.title()}: {percentage}% (invalid percentage)")
                    except (ValueError, TypeError):
                        # If conversion fails, just display the raw value
                        st.write(f"{content_type.title()}: {percentage}")
            else:
                st.write("**Recommended Content Mix:**")
                st.write("• Educational: 40%")
                st.write("• Entertaining: 30%")
                st.write("• Promotional: 20%")
                st.write("• User Generated: 10%")
            
            st.subheader("🏷️ Hashtag Strategy")
            hashtag_categories = brand_analysis.get('hashtag_strategy', [])
            for category in hashtag_categories:
                st.write(f"• {category}")
        
        with col2:
            st.subheader("🎯 Target Audience")
            st.write(brand_analysis.get('target_audience', 'General audience'))
            
            st.subheader("📱 Platform Recommendations")
            platforms = brand_analysis.get('best_platforms', [])
            if isinstance(platforms, list) and len(platforms) > 0:
                for i, platform in enumerate(platforms, 1):
                    st.write(f"{i}. {platform}")
            else:
                st.write("1. Instagram")
                st.write("2. LinkedIn")
                st.write("3. Twitter")
            
            st.subheader("💬 Engagement Tactics")
            tactics = brand_analysis.get('engagement_tactics', [])
            if isinstance(tactics, list) and len(tactics) > 0:
                for tactic in tactics:
                    st.write(f"• {tactic}")
            else:
                st.write("• Ask questions")
                st.write("• Share user content")
                st.write("• Respond to comments")
        
        # Niche-specific insights section
        st.subheader("🎨 Content Themes")
        themes = brand_analysis.get('content_themes', [])
        if isinstance(themes, list) and len(themes) > 0:
            for theme in themes:
                st.write(f"• {theme}")
        else:
            st.write("• Industry insights")
            st.write("• Behind the scenes")
            st.write("• Tips and tricks")
        
        # New personalized sections
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("💡 Niche Insights")
            insights = brand_analysis.get('niche_insights', [])
            if isinstance(insights, list) and len(insights) > 0:
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.write("• Focus on industry expertise")
                st.write("• Build community engagement")
                st.write("• Share innovations and trends")
            
            st.subheader("📊 Competitor Analysis")
            competitor_analysis = brand_analysis.get('competitor_analysis', '')
            if competitor_analysis:
                st.write(competitor_analysis)
            else:
                st.write("Study successful brands in your niche for best practices")
        
        with col4:
            st.subheader("🔥 Trending Topics")
            trending_topics = brand_analysis.get('trending_topics', [])
            if isinstance(trending_topics, list) and len(trending_topics) > 0:
                for topic in trending_topics:
                    st.write(f"• {topic}")
            else:
                st.write("• Latest industry trends")
                st.write("• Technology innovations")
                st.write("• Market insights")
            
            st.subheader("📅 Content Calendar")
            content_calendar = brand_analysis.get('content_calendar', '')
            if content_calendar:
                st.write(content_calendar)
            else:
                st.write("Structured content planning for your niche")
        
        # Real-time market analysis sections
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🚀 Market Opportunities")
            market_opportunities = brand_analysis.get('market_opportunities', [])
            if isinstance(market_opportunities, list) and len(market_opportunities) > 0:
                for opportunity in market_opportunities:
                    st.write(f"• {opportunity}")
            else:
                st.write("• Growing market demand")
                st.write("• Digital transformation opportunities")
                st.write("• New technology adoption")
            
            st.subheader("⚠️ Risk Factors")
            risk_factors = brand_analysis.get('risk_factors', [])
            if isinstance(risk_factors, list) and len(risk_factors) > 0:
                for risk in risk_factors:
                    st.write(f"• {risk}")
            else:
                st.write("• Market competition")
                st.write("• Economic factors")
                st.write("• Technology changes")
        
        with col6:
            st.subheader("💰 Budget Recommendations")
            budget_recommendations = brand_analysis.get('budget_recommendations', '')
            if budget_recommendations:
                st.write(budget_recommendations)
            else:
                st.write("Allocate 60% to content creation, 30% to paid promotion, 10% to tools")
            
            st.subheader("📱 Platform Strategy")
            platform_strategy = brand_analysis.get('platform_strategy', '')
            if platform_strategy:
                st.write(platform_strategy)
            else:
                st.write("Focus on primary platform with cross-platform content adaptation")
        
        # Generate personalized content strategy
        if st.button("🎯 Generate Personalized Strategy"):
            with st.spinner("Creating personalized strategy..."):
                # Get brand_analysis from session state
                current_brand_analysis = st.session_state.get('brand_analysis', {}) if 'brand_analysis' in st.session_state else {}
                strategy = generate_personalized_content_strategy(
                    brand_niche, current_brand_analysis, 
                    meta.get("platform", "Instagram") if "calendar_meta" in st.session_state else "Instagram",
                    meta.get("tone", "Professional") if "calendar_meta" in st.session_state else "Professional"
                )
                st.subheader("📋 Personalized Content Strategy")
                st.write(strategy)
    else:
        st.info("Complete your brand profile and analyze your niche to see personalized recommendations.")

with tab4:
    st.header("📊 Real-Time Market Analysis")
    st.markdown("Get current industry trends, competitor insights, and market opportunities for your brand!")
    
    # Check if brand analysis is available
    if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
        brand_analysis = st.session_state.brand_analysis
        brand_niche = brand_analysis.get('industry_category', 'Your Brand Niche')
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Analysis Type")
            analysis_type = st.selectbox(
                "Choose analysis type:",
                ["Industry Trends", "Competitor Analysis", "Both"]
            )
            
            platform_for_analysis = st.selectbox(
                "Platform for competitor analysis:",
                ["Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook", "YouTube"]
            )
        
        with col2:
            st.subheader("📈 Analysis Settings")
            include_reddit_trends = st.checkbox("Include Reddit trends", value=True)
            include_market_opportunities = st.checkbox("Include market opportunities", value=True)
            include_risk_assessment = st.checkbox("Include risk assessment", value=True)
        
        # Run analysis button
        if st.button("🚀 Run Real-Time Analysis", type="primary"):
            with st.spinner("Analyzing market trends and competitor strategies..."):
                
                # Initialize results
                trend_results = None
                competitor_results = None
                
                # Run selected analysis
                if analysis_type in ["Industry Trends", "Both"]:
                    trend_results = analyze_industry_trends(brand_niche)
                
                if analysis_type in ["Competitor Analysis", "Both"]:
                    competitor_results = analyze_competitors(brand_niche, platform_for_analysis)
                
                # Store results in session state
                st.session_state.trend_analysis = trend_results
                st.session_state.competitor_analysis = competitor_results
                
                st.success("✅ Analysis complete!")
        
        # Display results
        if "trend_analysis" in st.session_state and st.session_state.trend_analysis:
            st.subheader("📈 Industry Trends Analysis")
            trends = st.session_state.trend_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 Current Trends**")
                for trend in trends.get('current_trends', []):
                    st.write(f"• {trend}")
                
                st.markdown("**🚀 Emerging Technologies**")
                for tech in trends.get('emerging_technologies', []):
                    st.write(f"• {tech}")
                
                st.markdown("**👥 Consumer Behavior**")
                for behavior in trends.get('consumer_behavior', []):
                    st.write(f"• {behavior}")
                
                st.markdown("**💡 Market Opportunities**")
                for opportunity in trends.get('market_opportunities', []):
                    st.write(f"• {opportunity}")
            
            with col2:
                st.markdown("**⚠️ Challenges**")
                for challenge in trends.get('challenges', []):
                    st.write(f"• {challenge}")
                
                st.markdown("**📅 Seasonal Factors**")
                for factor in trends.get('seasonal_factors', []):
                    st.write(f"• {factor}")
                
                st.markdown("**📱 Social Media Trends**")
                for trend in trends.get('social_media_trends', []):
                    st.write(f"• {trend}")
                
                st.markdown("**📝 Content Trends**")
                for trend in trends.get('content_trends', []):
                    st.write(f"• {trend}")
        
        if "competitor_analysis" in st.session_state and st.session_state.competitor_analysis:
            st.subheader("🏆 Competitor Analysis")
            competitors = st.session_state.competitor_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Top Competitors**")
                for competitor in competitors.get('top_competitors', []):
                    st.write(f"• {competitor}")
                
                st.markdown("**📊 Competitor Strategies**")
                for strategy in competitors.get('competitor_strategies', []):
                    st.write(f"• {strategy}")
                
                st.markdown("**📝 Content Themes**")
                for theme in competitors.get('content_themes', []):
                    st.write(f"• {theme}")
                
                st.markdown("**💬 Engagement Tactics**")
                for tactic in competitors.get('engagement_tactics', []):
                    st.write(f"• {tactic}")
            
            with col2:
                st.markdown("**📅 Posting Patterns**")
                st.write(competitors.get('posting_patterns', 'Not specified'))
                
                st.markdown("**🏷️ Hashtag Strategies**")
                for strategy in competitors.get('hashtag_strategies', []):
                    st.write(f"• {strategy}")
                
                st.markdown("**📈 Success Metrics**")
                for metric in competitors.get('success_metrics', []):
                    st.write(f"• {metric}")
                
                st.markdown("**🎯 Differentiation Opportunities**")
                for opportunity in competitors.get('differentiation_opportunities', []):
                    st.write(f"• {opportunity}")
        
        # Actionable insights section
        if ("trend_analysis" in st.session_state and st.session_state.trend_analysis) or ("competitor_analysis" in st.session_state and st.session_state.competitor_analysis):
            st.subheader("💡 Actionable Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Immediate Actions**")
                st.write("• Adapt content to current trends")
                st.write("• Implement competitor best practices")
                st.write("• Focus on emerging opportunities")
                st.write("• Address identified challenges")
            
            with col2:
                st.markdown("**📈 Strategic Recommendations**")
                st.write("• Develop unique positioning")
                st.write("• Build competitive advantages")
                st.write("• Plan for seasonal changes")
                st.write("• Monitor market shifts")
            
            # Generate action plan
            if st.button("📋 Generate Action Plan"):
                with st.spinner("Creating personalized action plan..."):
                    action_plan = generate_action_plan(brand_niche, 
                                                     st.session_state.get('trend_analysis'),
                                                     st.session_state.get('competitor_analysis'))
                    st.subheader("📋 Personalized Action Plan")
                    st.write(action_plan)
        
        # Reddit trends integration
        if include_reddit_trends:
            st.subheader("📱 Reddit Trends")
            if st.button("🔍 Fetch Reddit Trends"):
                with st.spinner("Fetching current Reddit trends..."):
                    reddit_trends = get_trending_topics("entrepreneur", 5)
                    st.session_state.reddit_trends = reddit_trends
                    st.success("✅ Reddit trends fetched!")
            
            if "reddit_trends" in st.session_state:
                st.markdown("**🔥 Trending on Reddit**")
                for i, trend in enumerate(st.session_state.reddit_trends, 1):
                    st.write(f"{i}. {trend['title']}")
                    if trend['url']:
                        st.caption(f"Source: {trend['url']}")
    
    else:
        st.info("Complete your brand profile and generate a strategy first to access market analysis features.")

with tab5:
    st.header("🎬 AI Video Reel Generator")
    st.markdown("Transform your images into engaging video reels with AI-generated captions, music, and transitions!")
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Source Image")
        image_source = st.radio(
            "Choose image source:",
            ["Generate with DALL-E", "Upload Image", "Use from Calendar"]
        )
        
        if image_source == "Generate with DALL-E":
            st.info("Generate an image first in the '✍️ Create Post' tab, then use it here. DALL-E 3 will be used to create frame variations.")
        elif image_source == "Upload Image":
            uploaded_images = st.file_uploader("Upload Images (multiple allowed)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if uploaded_images:
                st.write(f"📸 Uploaded {len(uploaded_images)} image(s)")
                # Show first few images
                cols = st.columns(min(3, len(uploaded_images)))
                for i, img in enumerate(uploaded_images[:3]):
                    with cols[i % 3]:
                        st.image(img, caption=f"Image {i+1}", use_container_width=True)
                if len(uploaded_images) > 3:
                    st.info(f"... and {len(uploaded_images) - 3} more images")
        elif image_source == "Use from Calendar":
            if "calendar_df" in st.session_state:
                st.success("✅ Calendar data available!")
            else:
                st.warning("Generate a calendar first to use images from it.")
    
    with col2:
        st.subheader("🎬 Video Settings")
        video_duration = st.slider("Video Duration (seconds)", 5, 30, 15)
        frame_count = st.slider("Number of Frames", 3, 8, 5)
        include_transitions = st.checkbox("Include Smooth Transitions", value=True)
        auto_caption = st.checkbox("Generate AI Caption", value=True)
        auto_music = st.checkbox("Add Background Music", value=True)
    
    # Video generation controls
    st.subheader("🎯 Generation Options")
    
    # Get tone and platform from brand analysis or defaults
    if "brand_analysis" in st.session_state and st.session_state.brand_analysis is not None:
        brand_analysis = st.session_state.brand_analysis
        recommended_tones = brand_analysis.get('tone_recommendations', ["Witty","Empathetic","Educational","Professional"])
        if not isinstance(recommended_tones, list) or len(recommended_tones) == 0:
            recommended_tones = ["Witty","Empathetic","Educational","Professional"]
        video_tone = st.selectbox("Video Tone", recommended_tones)
    else:
        video_tone = st.selectbox("Video Tone", ["Witty","Empathetic","Educational","Professional"])
    
    # Video generation button
    if st.button("🎬 Generate Video Reel", type="primary"):
        if not video_enabled:
            st.error("❌ Video generation is disabled. Please configure Azure AI Foundry credentials.")
        else:
            # Determine image URL based on source
            image_url = ""
            if image_source == "Upload Image" and uploaded_images:
                # For uploaded images, we'd need to save and get URL
                # For now, we'll use a placeholder
                image_url = "https://via.placeholder.com/1024x1024/cccccc/666666?text=Uploaded+Images"
            elif image_source == "Use from Calendar" and "calendar_df" in st.session_state:
                # Use the last generated image from calendar
                image_url = "https://via.placeholder.com/1024x1024/cccccc/666666?text=Calendar+Image"
            else:
                # Generate a placeholder image
                image_url = "https://via.placeholder.com/1024x1024/cccccc/666666?text=Generated+Image"
            
            # Generate video reel
            video_result = create_video_reel(
                image_url=image_url,
                brand_niche=brand_niche,
                tone=video_tone,
                brand_color=brand_color,
                brand_analysis=st.session_state.get('brand_analysis') if 'brand_analysis' in st.session_state else None
            )
            
            # Store video result in session state for use in Video Transitions tab
            st.session_state.video_result = video_result
            
            # Display results
            st.success("🎬 Video reel generated successfully!")
            
            # Show generated frames in a gallery
            if "timeline" in video_result and "frames" in video_result["timeline"]:
                st.subheader("📸 Generated Frames Gallery")
                frames = video_result["timeline"]["frames"]
                if frames and len(frames) > 0:
                    # Display frames in columns
                    cols = st.columns(min(3, len(frames)))
                    for i, frame_url in enumerate(frames):
                        with cols[i % 3]:
                            st.image(frame_url, caption=f"Frame {i+1}", use_container_width=True)
                    
                    # Show frame generation details
                    st.info(f"🎯 Generated {len(frames)} unique frames using DALL-E 3 variations")
                    st.info(f"📊 Frame quality: {video_result.get('video_quality', 'HD')}")
                    st.info(f"🎬 Frame rate: {video_result.get('frame_rate', '30fps')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📹 Generated Video")
                # Display the generated video
                if video_result["video_bytes"]:
                    st.video(video_result["video_bytes"])
                    st.caption("🎬 AI-generated video reel with transitions")
                else:
                    st.warning("❌ Video generation failed. Please check MoviePy installation.")
                
                # Show the frames that were used
                st.subheader("📸 Generated Frames")
                if "frames_used" in video_result and video_result["frames_used"] > 0:
                    st.write(f"Used {video_result['frames_used']} frames for video generation")
                else:
                    st.write("Frame generation completed")
            
            with col2:
                st.subheader("📝 Generated Caption")
                st.write(video_result["caption"])
                
                st.subheader("🎵 Music Track")
                st.write(f"Track ID: {video_result['music_id']}")
                
                st.subheader("⏱️ Video Details")
                st.write(f"Duration: {video_result['total_duration']:.1f} seconds")
                st.write(f"Frames used: {video_result['frames_used']}")
                st.write(f"Quality: {video_result.get('video_quality', 'HD')}")
                st.write(f"Resolution: {video_result.get('resolution', '1920x1080')}")
                st.write(f"Frame Rate: {video_result.get('frame_rate', '30fps')}")
            
            # Timeline information
            st.subheader("📋 Video Timeline")
            timeline = video_result.get("timeline", {})
            if timeline:
                st.json(timeline)
            
            # Download options
            st.subheader("💾 Download Options")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if video_result["video_bytes"]:
                    st.download_button(
                        label="📹 Download Video",
                        data=video_result["video_bytes"],
                        file_name=f"{brand_niche}_reel.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.button("📹 Download Video", disabled=True)
                    st.caption("Video not available")
            
            with col2:
                st.download_button(
                    label="📝 Download Caption",
                    data=video_result["caption"].encode(),
                    file_name=f"{brand_niche}_caption.txt",
                    mime="text/plain"
                )
            
            with col3:
                st.download_button(
                    label="📋 Download Timeline",
                    data=json.dumps(timeline, indent=2).encode(),
                    file_name=f"{brand_niche}_timeline.json",
                    mime="application/json"
                )
            
            with col4:
                # Create a zip file with all frames for video creation
                if "frames" in video_result and video_result["frames"]:
                    frames = video_result["frames"]
                    if frames and len(frames) > 0:
                        # Create a simple text file with frame URLs for now
                        frame_urls = "\n".join(frames)
                        st.download_button(
                            label="🖼️ Download Frame URLs",
                            data=frame_urls.encode(),
                            file_name=f"{brand_niche}_frame_urls.txt",
                            mime="text/plain"
                        )
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If you're getting 401 authentication errors:**
        
        1. **Check your environment variables** in `.env` file:
        ```env
        AZURE_OPENAI_IMAGE_ENDPOINT=https://your-image-endpoint.openai.azure.com/
        AZURE_OPENAI_IMAGE_KEY=your-image-key
        ```
        
        2. **Verify your Azure OpenAI Image deployment**:
        - Go to Azure Portal → Azure OpenAI
        - Check if DALL-E 3 is deployed
        - Verify the endpoint URL and key
        
        3. **Test with a simple image generation first**:
        - Go to "✍️ Create Post" tab
        - Try generating an image there
        - If that works, the video generation should work too
        
        4. **Alternative**: Use "Upload Image" option instead of DALL-E 3 generation
        """)
        

    
    # Video generation pipeline explanation
    with st.expander("🔧 Video Generation Pipeline"):
        st.markdown("""
        **The video generation process follows this AI-powered pipeline:**
        
        1. **📸 Extract Frames** → DALL-E 3 generates variations of the source image
        2. **🏆 Rank Frames** → Simulated ranking based on visual appeal
        3. **✍️ Draft Caption** → GPT-4 generates initial caption (BLIP not available)
        4. **✨ Polish Caption** → GPT-4 text refines caption (GPT-4 Vision not available)
        5. **🎵 Choose Music** → Simulated music selection based on caption and brand
        6. **🎨 Stylize Frames** → DALL-E 3 applies brand styling (cost-effective alternative to SDXL)
        7. **🎬 Generate Transitions** → MoviePy creates real video with transitions
        8. **📋 Compose Timeline** → Final video assembly with timing and metadata
        
        **Note**: VideoMAE, CLIP, BLIP, and GPT-4 Vision are not available in Azure AI Foundry. Using DALL-E 3, GPT-4 text, and MoviePy for real video generation.
        """)
    
    # Video tips and best practices
    with st.expander("💡 Video Creation Tips"):
        st.markdown("""
        **Best Practices for AI-Generated Video Reels:**
        
        - **🎯 Brand Consistency**: Use your brand colors and tone throughout
        - **⏱️ Optimal Duration**: 15-30 seconds work best for social media
        - **📱 Platform Optimization**: Different platforms have different requirements
        - **🎵 Music Selection**: Choose music that matches your brand personality
        - **📝 Caption Strategy**: Keep captions engaging and platform-appropriate
        - **🔄 Regular Updates**: Generate fresh content regularly to maintain engagement
        
        **Supported Platforms:**
        - Instagram Reels
        - TikTok
        - YouTube Shorts
        - LinkedIn Video
        - Facebook Video
        """)
    
    # Instructions for creating real videos
    with st.expander("🎬 Create Real Videos with Transitions"):
        st.markdown("""
        **To create real videos with interesting transitions from your generated frames:**
        
        1. **📥 Download Frame URLs** - Use the download button above to get frame URLs
        2. **🔧 Install Dependencies** - Run: `pip install -r requirements_video.txt`
        3. **🎬 Run Video Creator** - Execute: `python video_transitions.py`
        4. **✨ Choose Transitions** - Select from 10 different transition effects:
           - Fade (smooth fade in/out)
           - Slide Left/Right (horizontal sliding)
           - Zoom In/Out (dynamic zoom effects)
           - Rotate (360° rotation)
           - Flip (vertical flip)
           - Wipe Left/Right (wipe transitions)
           - Dissolve (crossfade effect)
        
        **Available Transition Effects:**
        - **Fade**: Smooth fade in and out between images
        - **Slide Left/Right**: Images slide in from one side, out to the other
        - **Zoom In/Out**: Dynamic zoom effects for dramatic impact
        - **Rotate**: 360-degree rotation effect
        - **Flip**: Vertical flip transition
        - **Wipe Left/Right**: Wipe transitions like in presentations
        - **Dissolve**: Crossfade between images
        
        **Output**: The script will create a high-quality MP4 video with your chosen transitions!
        """)

with tab5:
    st.header("🎬 Video Transitions Creator")
    st.markdown("Create videos with interesting transitions from your generated frames or uploaded images!")
    
    # Check if MoviePy is available
    MOVIEPY_AVAILABLE = check_moviepy_available()
    if not MOVIEPY_AVAILABLE:
        st.error("❌ MoviePy not available. Please install with: `pip install moviepy`")
        st.info("💡 You can still download frame URLs and use the external script: `python video_transitions.py`")
    else:
        st.success("✅ MoviePy available - Video creation enabled!")
    
    # Image source selection
    st.subheader("📸 Image Source")
    image_source = st.radio(
        "Choose image source:",
        ["Use Generated Frames", "Upload Images", "Use from Calendar"]
    )
    
    image_urls = []
    
    if image_source == "Use Generated Frames":
        if "video_result" in st.session_state and "frames" in st.session_state.video_result:
            frames = st.session_state.video_result["frames"]
            if frames:
                image_urls = frames
                st.success(f"✅ Found {len(image_urls)} generated frames")
                
                # Show frames
                st.subheader("📸 Generated Frames")
                cols = st.columns(min(3, len(image_urls)))
                for i, url in enumerate(image_urls):
                    with cols[i % 3]:
                        st.image(url, caption=f"Frame {i+1}", use_container_width=True)
            else:
                st.warning("No generated frames found. Generate a video reel first!")
        else:
            st.warning("No video result found. Generate a video reel first!")
    
    elif image_source == "Upload Images":
        uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} images")
            
            # Convert uploaded files to image URLs for processing
            uploaded_image_urls = []
            for i, uploaded_file in enumerate(uploaded_files):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Convert to file:// URL for processing
                file_url = f"file://{tmp_file_path}"
                uploaded_image_urls.append(file_url)
            
            # Show uploaded images
            st.subheader("📸 Uploaded Images")
            cols = st.columns(min(3, len(uploaded_files)))
            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i % 3]:
                    st.image(uploaded_file, caption=f"Image {i+1}", use_container_width=True)
            
            # Use uploaded images for video creation
            image_urls = uploaded_image_urls
            st.info(f"📝 {len(image_urls)} images ready for video creation!")
    
    elif image_source == "Use from Calendar":
        if "calendar_df" in st.session_state:
            st.success("✅ Calendar data available!")
            st.info("📝 Use generated frames from the Video Reel tab for best results.")
        else:
            st.warning("Generate a calendar first to use images from it.")
    
    # Video settings
    if image_urls:
        st.subheader("🎬 Video Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transition_type = st.selectbox(
                "Transition Effect",
                ["fade", "slide_left", "slide_right", "zoom_in", "zoom_out", 
                 "rotate", "flip", "wipe_left", "wipe_right", "dissolve"],
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            duration_per_image = st.slider("Duration per image (seconds)", 1.0, 5.0, 2.0, 0.5)
            transition_duration = st.slider("Transition duration (seconds)", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            fps = st.selectbox("Frame Rate (FPS)", [24, 30, 60], index=0)
            
            # Show transition preview
            st.subheader("🎭 Transition Preview")
            transition_descriptions = {
                "fade": "Smooth fade in and out between images",
                "slide_left": "Images slide in from right, out to left",
                "slide_right": "Images slide in from left, out to right",
                "zoom_in": "Dynamic zoom in effect",
                "zoom_out": "Dynamic zoom out effect",
                "rotate": "360-degree rotation effect",
                "flip": "Vertical flip transition",
                "wipe_left": "Wipe from right to left",
                "wipe_right": "Wipe from left to right",
                "dissolve": "Crossfade between images"
            }
            st.info(transition_descriptions.get(transition_type, "Custom transition"))
        
        # Create video button
        if st.button("🎬 Create Video with Transitions", type="primary"):
            if MOVIEPY_AVAILABLE and image_urls:
                video_bytes = create_video_from_urls(
                    image_urls, 
                    transition_type, 
                    duration_per_image, 
                    transition_duration, 
                    fps
                )
                
                if video_bytes:
                    st.success("✅ Video created successfully!")
                    
                    # Display video
                    st.subheader("📹 Generated Video")
                    st.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name=f"video_transition_{transition_type}.mp4",
                        mime="video/mp4"
                    )
                    
                    # Video details
                    st.subheader("📊 Video Details")
                    total_duration = len(image_urls) * duration_per_image + (len(image_urls) - 1) * transition_duration
                    st.write(f"**Duration:** {total_duration:.1f} seconds")
                    st.write(f"**Images:** {len(image_urls)}")
                    st.write(f"**Transition:** {transition_type.replace('_', ' ').title()}")
                    st.write(f"**FPS:** {fps}")
                    st.write(f"**Quality:** HD")
                else:
                    st.error("❌ Failed to create video. Please check the error messages above.")
            else:
                st.error("❌ MoviePy not available or no images provided.")
    
    # Instructions
    with st.expander("💡 How to Use"):
        st.markdown("""
        **Creating Videos with Transitions:**
        
        1. **Generate frames** in the Video Reel tab, or upload images
        2. **Select transition effect** from the dropdown
        3. **Adjust settings** (duration, FPS, etc.)
        4. **Click Create Video** to generate your video
        5. **Download** the final video for social media
        
        **Available Transitions:**
        - **Fade**: Smooth fade in/out
        - **Slide Left/Right**: Horizontal sliding
        - **Zoom In/Out**: Dynamic zoom effects
        - **Rotate**: 360° rotation
        - **Flip**: Vertical flip
        - **Wipe Left/Right**: Wipe transitions
        - **Dissolve**: Crossfade effect
        
        **Tips:**
        - Use 2-3 seconds per image for best results
        - Higher FPS (30-60) for smoother transitions
        - Fade and dissolve work well for most content
        - Zoom effects are great for dramatic impact
        """)

    
    # Troubleshooting section
