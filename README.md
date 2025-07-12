# 📊 PostCraft Pro: AI-Powered Social Media Content Creator

A sophisticated Streamlit application that generates personalized social media content using Azure OpenAI services, Reddit trend analysis, and brand-specific strategies.

## 🚀 Features

### 🎯 **Brand Niche Personalization**
- **AI Brand Analysis**: Analyzes your brand niche and provides personalized recommendations
- **Industry-Specific Content**: Tailored content strategies for different industries
- **Smart Subreddit Mapping**: Automatically suggests relevant subreddits based on your brand
- **Personalized Tone Recommendations**: Suggests appropriate tones for your industry

### 📅 **Trend-Driven Calendar**
- **14-Day Content Calendar**: Generate content plans based on trending Reddit posts
- **Real-time Trend Analysis**: Fetches current viral posts from relevant subreddits
- **Platform Optimization**: Tailored strategies for Instagram, LinkedIn, and Twitter
- **Smart Fallback System**: Graceful handling when subreddits are unavailable

### ✍️ **AI Content Generation**
- **Personalized Post Creation**: Brand-aware content generation
- **DALL-E Image Generation**: AI-generated images matching your brand aesthetic
- **Hashtag Strategy**: Industry-specific hashtag recommendations
- **Engagement Tactics**: Niche-specific engagement strategies

### 🎬 **AI Video Reel Generation**
- **Multi-Model Pipeline**: Complete video generation using Azure AI Foundry models
- **Frame Extraction & Ranking**: VideoMAE and CLIP-ViT-L/14 for optimal frame selection
- **AI Caption Generation**: BLIP-2-Flan-T5-XXL and GPT-4 Vision for engaging captions
- **Smart Music Selection**: Vector search for copyright-free music matching your content
- **Style Transfer**: SDXL Style Transfer LoRA for brand-consistent styling
- **Smooth Transitions**: Stable Video Diffusion for professional video transitions
- **Multi-Platform Support**: Optimized for Instagram Reels, TikTok, YouTube Shorts, and more

### 🎨 **Brand Strategy Dashboard**
- **Content Mix Visualization**: Recommended content distribution
- **Platform Rankings**: Best platforms for your niche
- **Target Audience Analysis**: Detailed audience insights
- **Personalized Strategy Generator**: Comprehensive content strategies

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Reddit API access (optional, for enhanced functionality)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/postcraft-pro.git
   cd postcraft-pro
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas openai python-dotenv requests
   ```

3. **Environment Configuration**
   Create a `.env` file with your Azure OpenAI credentials:
   ```env
   # Azure OpenAI Text Service
   AZURE_OPENAI_TEXT_ENDPOINT=your_text_endpoint
   AZURE_OPENAI_TEXT_KEY=your_text_api_key
   
   # Azure OpenAI Image Service (DALL-E)
   AZURE_OPENAI_IMAGE_ENDPOINT=your_image_endpoint
   AZURE_OPENAI_IMAGE_KEY=your_image_api_key
   
   # Azure AI Foundry Video Generation (Optional)
   AZURE_AI_FOUNDRY_ENDPOINT=your_ai_foundry_endpoint
   AZURE_AI_FOUNDRY_API_KEY=your_ai_foundry_api_key
   
   # Reddit User Agent (optional)
   REDDIT_USER_AGENT=PostCraftPro/1.0
   ```

4. **Run the application**
   ```bash
   streamlit run postcraft_azure_ready.py
   ```

## 🎯 Usage

### 1. **Brand Profile Setup**
- Enter your brand niche/industry
- Provide a detailed brand description
- Upload your brand logo (optional)
- Set your brand's primary color

### 2. **Brand Analysis**
- Click "🔍 Analyze Brand Niche" to get personalized recommendations
- Review industry-specific insights and strategies

### 3. **Content Calendar Generation**
- Select your target platform (Instagram, LinkedIn, Twitter)
- Choose relevant subreddits for content inspiration
- Pick appropriate tone for your brand
- Generate a 14-day content calendar

### 4. **Post Creation**
- Select entries from your calendar
- Generate final posts with AI-generated images
- Download and schedule your content

### 5. **Video Reel Generation**
- Choose image source (DALL-E generated, uploaded, or from calendar)
- Configure video settings (duration, frames, transitions)
- Generate AI-powered video reels with captions and music
- Download videos, captions, and timeline data
- Optimize for different social media platforms

## 🔧 Configuration

### Supported Industries
- **Technology**: Programming, startups, AI/ML
- **Fitness**: Health, nutrition, wellness
- **Food**: Recipes, cooking, meal prep
- **Fashion**: Style, trends, lifestyle
- **Business**: Entrepreneurship, marketing, sales
- **Education**: Science, learning, knowledge sharing
- **Health**: Mental health, wellness, medical
- **Travel**: Backpacking, digital nomad, photography
- **Finance**: Personal finance, investing, crypto
- **Entertainment**: Movies, music, gaming
- **Sports**: Athletics, fitness, team sports
- **Lifestyle**: Productivity, minimalism, self-improvement
- **Beauty**: Skincare, makeup, hair care
- **Parenting**: Family, children, pregnancy
- **Pets**: Animals, pet care, training

### Platform Optimizations
- **Instagram**: Visual content, stories, reels
- **LinkedIn**: Professional networking, thought leadership
- **Twitter**: Real-time updates, conversations, trends

### Video Generation Pipeline
The AI video generation follows an 8-step pipeline using Azure AI Foundry models:

1. **Frame Extraction** → VideoMAE model analyzes images and extracts key frames
2. **Frame Ranking** → CLIP-ViT-L/14 ranks frames by visual appeal (cosine > 0.28)
3. **Caption Drafting** → BLIP-2-Flan-T5-XXL generates initial captions
4. **Caption Polishing** → GPT-4 Vision refines captions with frame context
5. **Music Selection** → Vector search finds copyright-free music matching content
6. **Style Transfer** → SDXL Style Transfer LoRA applies brand styling
7. **Transition Generation** → Stable Video Diffusion creates smooth transitions
8. **Timeline Composition** → Final video assembly with timing and metadata

### Supported Video Platforms
- **Instagram Reels**: 9:16 aspect ratio, 15-60 seconds
- **TikTok**: 9:16 aspect ratio, 15-60 seconds
- **YouTube Shorts**: 9:16 aspect ratio, up to 60 seconds
- **LinkedIn Video**: 1:1 or 16:9 aspect ratio, up to 10 minutes
- **Facebook Video**: Various aspect ratios, up to 240 minutes

## 🛡️ Security

- **Environment Variables**: All API keys stored securely in `.env` file
- **Git Ignore**: Sensitive files automatically excluded from version control
- **Error Handling**: Graceful handling of API failures and network issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Azure OpenAI** for powerful AI capabilities
- **Streamlit** for the amazing web app framework
- **Reddit** for trending content inspiration
- **DALL-E** for AI-generated images

## 📞 Support

If you encounter any issues or have questions:
- Create an issue on GitHub
- Check the troubleshooting section below
- Review the error logs in the Streamlit interface

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all Azure OpenAI credentials are correctly set in `.env`
   - Verify endpoint URLs are correct
   - Check API key permissions

2. **Subreddit Fetching Issues**
   - Some subreddits may be private or restricted
   - The app will automatically use fallback subreddits
   - Check the warning messages for details

3. **Image Generation Failures**
   - Ensure DALL-E API is properly configured
   - Check image generation quotas
   - Verify brand color format (hex codes)

4. **Performance Issues**
   - Reduce the number of days in calendar generation
   - Use fewer subreddits for faster processing
   - Check your internet connection

5. **Video Generation Issues**
   - Ensure Azure AI Foundry credentials are properly configured
   - Check video generation quotas and model availability
   - Verify image upload formats (PNG, JPG, JPEG)
   - Ensure sufficient storage for video processing

---

**Made with ❤️ for social media creators and marketers** 