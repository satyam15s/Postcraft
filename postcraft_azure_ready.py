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

# Load environment variables
load_dotenv()

# ——— Azure OpenAI setup ———
TEXT_ENDPOINT    = os.getenv("AZURE_OPENAI_TEXT_ENDPOINT")
TEXT_API_VERSION = "2023-05-15"
TEXT_KEY         = os.getenv("AZURE_OPENAI_TEXT_KEY")

openai.api_type    = "azure"
openai.api_base    = TEXT_ENDPOINT
openai.api_version = TEXT_API_VERSION
openai.api_key     = TEXT_KEY

text_client = AzureOpenAI(
    api_key       = TEXT_KEY,
    api_version   = TEXT_API_VERSION,
    azure_endpoint= TEXT_ENDPOINT
)

# ——— Image (DALL·E) setup ———
IMAGE_ENDPOINT    = os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
IMAGE_API_VERSION = "2024-02-01"
IMAGE_KEY         = os.getenv("AZURE_OPENAI_IMAGE_KEY")

image_client = AzureOpenAI(
    api_key       = IMAGE_KEY,
    api_version   = IMAGE_API_VERSION,
    azure_endpoint= IMAGE_ENDPOINT
)

# ——— Brand Niche Analysis & Personalization ———
def analyze_brand_niche(brand_niche: str, brand_description: str = "") -> dict:
    """Analyze brand niche and provide personalized recommendations"""
    prompt = f"""
Analyze this brand niche and provide personalized recommendations:
Brand Niche: {brand_niche}
Brand Description: {brand_description}

Provide a JSON response with:
1. "industry_category": The main industry category
2. "target_audience": Primary target audience
3. "content_themes": 5-7 content themes that work well for this niche
4. "recommended_subreddits": 5-7 relevant subreddits for content inspiration
5. "best_platforms": Rank platforms by effectiveness (Instagram, LinkedIn, Twitter, TikTok, Facebook)
6. "tone_recommendations": 3-4 appropriate tones for this niche
7. "content_mix": Suggested content mix (educational %, entertaining %, promotional %, etc.)
8. "hashtag_strategy": Recommended hashtag categories
9. "posting_frequency": Optimal posting frequency
10. "engagement_tactics": Specific engagement tactics for this niche

Return only valid JSON.
"""
    
    try:
        resp = text_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=500
        )
        analysis = json.loads(resp.choices[0].message.content.strip())
        
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
        # Fallback analysis with guaranteed structure
        return {
            "industry_category": "General",
            "target_audience": "General audience",
            "content_themes": ["Industry insights", "Behind the scenes", "Tips and tricks", "Success stories", "Trending topics"],
            "recommended_subreddits": ["memes", "funny", "todayilearned", "explainlikeimfive", "science"],
            "best_platforms": ["Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook"],
            "tone_recommendations": ["Professional", "Educational", "Witty", "Empathetic"],
            "content_mix": {"educational": 40, "entertaining": 30, "promotional": 20, "user_generated": 10},
            "hashtag_strategy": ["Industry-specific", "Trending", "Branded", "Community"],
            "posting_frequency": "3-5 times per week",
            "engagement_tactics": ["Ask questions", "Share user content", "Respond to comments", "Create polls"]
        }

def get_niche_specific_subreddits(brand_niche: str) -> list:
    """Get relevant subreddits based on brand niche"""
    niche_subreddit_map = {
        "Technology": ["programming", "technology", "gadgets", "startups", "artificial", "MachineLearning"],
        "Fitness": ["fitness", "bodybuilding", "running", "yoga", "nutrition", "loseit"],
        "Food": ["food", "recipes", "cooking", "MealPrepSunday", "slowcooking", "baking"],
        "Fashion": ["fashion", "streetwear", "malefashionadvice", "femalefashionadvice", "sneakers"],
        "Business": ["entrepreneur", "smallbusiness", "marketing", "sales", "startups", "investing"],
        "Education": ["science", "todayilearned", "explainlikeimfive", "askscience", "education"],
        "Health": ["health", "nutrition", "fitness", "mentalhealth", "meditation", "wellness"],
        "Travel": ["travel", "backpacking", "digitalnomad", "solotravel", "travelphotography"],
        "Finance": ["personalfinance", "investing", "wallstreetbets", "financialindependence", "cryptocurrency"],
        "Entertainment": ["movies", "television", "music", "gaming", "books", "entertainment"],
        "Sports": ["sports", "soccer", "basketball", "nfl", "baseball", "fitness"],
        "Lifestyle": ["lifestyle", "productivity", "minimalism", "selfimprovement", "motivation"],
        "Beauty": ["beauty", "skincareaddiction", "makeupaddiction", "haircarescience", "beauty"],
        "Parenting": ["parenting", "mommit", "daddit", "toddlers", "pregnant"],
        "Pets": ["aww", "dogs", "cats", "pets", "dogtraining", "catcare"]
    }
    
    # Find best match for brand niche
    for category, subreddits in niche_subreddit_map.items():
        if category.lower() in brand_niche.lower():
            return subreddits
    
    # Default fallback
    return ["memes", "funny", "todayilearned", "explainlikeimfive", "science"]

def generate_personalized_content_strategy(brand_niche: str, brand_analysis: dict, platform: str, tone: str) -> str:
    """Generate personalized content strategy based on brand analysis"""
    prompt = f"""
You are a social media strategist creating personalized content for a {brand_niche} brand.

Brand Analysis:
- Industry: {brand_analysis.get('industry_category', 'General')}
- Target Audience: {brand_analysis.get('target_audience', 'General audience')}
- Content Themes: {', '.join(brand_analysis.get('content_themes', []))}
- Content Mix: {brand_analysis.get('content_mix', {})}
- Engagement Tactics: {', '.join(brand_analysis.get('engagement_tactics', []))}

Platform: {platform}
Tone: {tone}

Create a personalized content strategy that:
1. Aligns with the brand's industry and target audience
2. Uses the recommended content themes
3. Follows the suggested content mix
4. Incorporates platform-specific best practices
5. Maintains the specified tone
6. Includes engagement tactics for this niche

Provide a comprehensive strategy that feels authentic to this specific brand niche.
"""
    
    resp = text_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()

# ——— Fetch Reddit hot posts via public JSON (skip stickied) ———
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
    return resp.choices[0].message.content.strip()

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
    ).choices[0].message.content

    # Extract last "Image Prompt:" line
    final_prompt = next(
        line.replace("Image Prompt:", "").strip()
        for line in cot_resp.splitlines()
        if line.startswith("Image Prompt:")
    )

    if include_logo:
        final_prompt += ", include the brand logo"

    # 2) Call DALL·E with 1024×1024
    img = image_client.images.generate(
        model="dall-e-3",
        prompt=final_prompt,
        n=1,
        size="1024x1024"
    )
    return img.data[0].url, final_prompt

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
    return resp.choices[0].message.content.strip()

# ——— Streamlit UI ———
st.set_page_config(page_title="PostCraft Pro", layout="wide")
st.title("📊 PostCraft Pro: Trend Calendar & Post Creator")

# Enhanced Brand Profile
st.sidebar.header("🎯 Brand Profile")
brand_niche = st.sidebar.text_input("Brand Niche/Industry", "Your Brand Niche")
brand_description = st.sidebar.text_area("Brand Description", "Describe your brand, values, and target audience...", height=100)

# Analyze brand niche when description is provided
brand_analysis = None
if brand_description and brand_description != "Describe your brand, values, and target audience...":
    if st.sidebar.button("🔍 Analyze Brand Niche"):
        with st.spinner("Analyzing your brand niche..."):
            brand_analysis = analyze_brand_niche(brand_niche, brand_description)
            st.session_state.brand_analysis = brand_analysis
            st.sidebar.success("Brand analysis complete!")

# Display brand analysis if available
if "brand_analysis" in st.session_state:
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

tab1, tab2, tab3 = st.tabs(["📅 Calendar", "✍️ Create Post", "🎯 Brand Strategy"])

with tab1:
    st.header("14-Day Trend-Driven Calendar")
    
    # Enhanced platform selection with brand analysis
    if brand_analysis:
        recommended_platforms = brand_analysis.get('best_platforms', ["Instagram","LinkedIn","Twitter"])
        # Ensure it's a list
        if not isinstance(recommended_platforms, list) or len(recommended_platforms) == 0:
            recommended_platforms = ["Instagram","LinkedIn","Twitter"]
        platform = st.selectbox("Platform", recommended_platforms)
    else:
        platform = st.selectbox("Platform", ["Instagram","LinkedIn","Twitter"])
    
    # Niche-specific subreddit suggestions
    if brand_analysis:
        recommended_subreddits = brand_analysis.get('recommended_subreddits', ["memes"])
        # Ensure it's a list
        if not isinstance(recommended_subreddits, list) or len(recommended_subreddits) == 0:
            recommended_subreddits = ["memes"]
        subreddit = st.selectbox("Subreddit", recommended_subreddits)
    else:
        niche_subreddits = get_niche_specific_subreddits(brand_niche)
        subreddit = st.selectbox("Subreddit", niche_subreddits)
    
    # Enhanced tone selection
    if brand_analysis:
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
        trends = get_trending_topics(subreddit, days)
        rows = []
        
        for i, t in enumerate(trends):
            d = start + timedelta(days=i)
            strategy = generate_post_for_trend(subreddit, tone, platform, t, brand_niche, brand_analysis)
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
            "brand_analysis": brand_analysis
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
                st.image(img_url, use_column_width=True)
                st.caption(f"Prompt: `{img_prompt}`")
    else:
        st.info("Generate your calendar first on the 📅 Calendar tab.")

with tab3:
    st.header("🎯 Brand Strategy Dashboard")
    
    if brand_analysis:
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
        
        st.subheader("🎨 Content Themes")
        themes = brand_analysis.get('content_themes', [])
        if isinstance(themes, list) and len(themes) > 0:
            for theme in themes:
                st.write(f"• {theme}")
        else:
            st.write("• Industry insights")
            st.write("• Behind the scenes")
            st.write("• Tips and tricks")
        
        # Generate personalized content strategy
        if st.button("🎯 Generate Personalized Strategy"):
            with st.spinner("Creating personalized strategy..."):
                strategy = generate_personalized_content_strategy(
                    brand_niche, brand_analysis, 
                    meta.get("platform", "Instagram") if "calendar_meta" in st.session_state else "Instagram",
                    meta.get("tone", "Professional") if "calendar_meta" in st.session_state else "Professional"
                )
                st.subheader("📋 Personalized Content Strategy")
                st.write(strategy)
    else:
        st.info("Complete your brand profile and analyze your niche to see personalized recommendations.")
