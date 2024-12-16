import os
import requests
from datetime import datetime
from typing import List
import ollama
from math import log
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from pydantic import BaseModel


class YouTubeVideo(BaseModel):
    score: float
    title: str
    channel_name: str
    views: int
    likes: int
    subscribers: int
    days_since_published: int
    video_id: str
    url: str
    description: str


def search_youtube(topic: str, max_results: int = 5) -> List[YouTubeVideo]:
    """
    Search YouTube videos using the YouTube Data API v3

    Args:
        topic: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List[YouTubeVideo]: List of YouTubeVideo objects containing video information
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return "Error: YouTube API key not found in environment variables"

        # Search for videos
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "maxResults": max_results,
            "q": topic,
            "type": "video",
            "key": api_key,
        }

        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()

        if "items" not in search_data:
            return f"Error: No results found or API error: {search_data.get('error', {}).get('message', 'Unknown error')}"

        results = []
        for item in search_data["items"]:
            video_id = item["id"]["videoId"]

            # Get video statistics
            video_url = "https://www.googleapis.com/youtube/v3/videos"
            video_params = {
                "part": "statistics,snippet",
                "id": video_id,
                "key": api_key,
            }

            video_response = requests.get(video_url, params=video_params)
            video_data = video_response.json()

            if "items" in video_data and video_data["items"]:
                video_stats = video_data["items"][0]["statistics"]
                video_snippet = video_data["items"][0]["snippet"]

                # Get channel statistics
                channel_id = item["snippet"]["channelId"]
                channel_url = "https://www.googleapis.com/youtube/v3/channels"
                channel_params = {
                    "part": "statistics",
                    "id": channel_id,
                    "key": api_key,
                }

                channel_response = requests.get(channel_url, params=channel_params)
                channel_data = channel_response.json()

                # Calculate days since published
                publish_date = datetime.strptime(
                    video_snippet["publishedAt"][:10], "%Y-%m-%d"
                )
                days_since_published = (datetime.now() - publish_date).days

                view_count = video_stats.get("viewCount", "0")
                like_count = video_stats.get("likeCount", "0")
                subscriber_count = (
                    channel_data["items"][0]["statistics"].get("subscriberCount", "0")
                    if "items" in channel_data
                    else "0"
                )

                video = YouTubeVideo(
                    score=calculate_score(
                        days_since_published,
                        int(view_count),
                        int(subscriber_count),
                        int(like_count),
                    ),
                    title=video_snippet["title"],
                    channel_name=video_snippet["channelTitle"],
                    views=int(view_count),
                    likes=int(like_count),
                    subscribers=int(subscriber_count),
                    days_since_published=days_since_published,
                    video_id=video_id,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    description=video_snippet["description"][:200],
                )
                results.append(video)

        return results

    except Exception as e:
        raise Exception(f"Error searching YouTube: {str(e)}")


def calculate_score(
    days_since_published: int, views: int, subscribers: int, likes: int
) -> float:
    """
    Calculate video score based on multiple metrics

    Args:
        days_since_published: Number of days since video was published
        views: Number of video views
        subscribers: Number of channel subscribers
        likes: Number of video likes

    Returns:
        float: Calculated score between 0 and 1
    """
    # Convert metrics to numbers, use 0 if N/A
    views = int(views) if str(views).isdigit() else 0
    subscribers = int(subscribers) if str(subscribers).isdigit() else 0
    likes = int(likes) if str(likes).isdigit() else 0

    # Normalize dates (newer = better, max age considered is 365 days)
    date_score = max(0, (365 - min(days_since_published, 365)) / 365)

    # Normalize other metrics using log scale to handle large numbers
    # Add 1 to avoid log(0)
    view_score = min(1, log(views + 1) / log(10000000))  # Assuming 10M views is max
    subscriber_score = min(
        1, log(subscribers + 1) / log(10000000)
    )  # Assuming 10M subs is max
    like_score = min(1, log(likes + 1) / log(100000))  # Assuming 100K likes is max

    # Calculate weighted score
    final_score = (
        0.5 * date_score + 0.3 * view_score + 0.1 * subscriber_score + 0.1 * like_score
    )

    return round(final_score, 3)


def get_video_transcript(video_url: str) -> str:
    """
    Get the transcript/subtitles from a YouTube video

    Args:
        video_url: Full YouTube video URL or video ID

    Returns:
        str: Formatted transcript text or error message
    """
    try:
        # Extract video ID from URL
        if "youtube.com" in video_url or "youtu.be" in video_url:
            if "youtube.com" in video_url:
                query = urlparse(video_url).query
                video_id = parse_qs(query).get("v", [None])[0]
            else:  # youtu.be
                video_id = urlparse(video_url).path[1:]
        else:
            # Assume the input is directly a video ID
            video_id = video_url

        if not video_id:
            return "Error: Could not extract video ID from URL"

        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Format transcript text
        formatted_transcript = ""
        for entry in transcript_list:
            formatted_transcript += f"{entry['text']}\n"

        return formatted_transcript

    except Exception as e:
        return f"Error getting transcript: {str(e)}"


def generate_video_summary(video: YouTubeVideo, topic: str) -> str:
    transcript = get_video_transcript(video.url)
    summary = create_summary(transcript, topic)
    return summary


def create_summary(text: str, topic: str, extra_instructions: str = "") -> str:
    prompt = f"""
    <transcript>
    {text}
    </transcript>
    You are a helpful assistant that summarizes YouTube videos. The audience is people looking
    to learn about a {topic}. 
    You are required to create a comprehensive summary of the video. Mind the details they are important.
    The summary should be in a way that is easy to understand and follow.
    
    {extra_instructions}
    """
    summary = ollama.generate(model="llama3.1:8b", prompt=prompt)
    return summary.response


def summarize_videos(videos: List[YouTubeVideo], topic: str) -> str:
    summaries = []
    for video in videos:
        summary = generate_video_summary(video, topic)
        summaries.append(summary)
    return create_summary(
        str(summaries),
        topic,
        extra_instructions="The summaries are from different videos. Combine them into a single summary.",
    )


if __name__ == "__main__":
    # Test the function with a sample search
    test_topic = "open ai recent announcements"
    print("Searching YouTube for:", test_topic)
    videos = search_youtube(test_topic, max_results=5)
    summary = summarize_videos(videos, test_topic)
    with open(f"{test_topic.replace(' ', '_')}_summary.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(summary)

    # # Access structured data
    # for video in videos:
    #     print(f"Title: {video.title}")
    #     print(f"Score: {video.score}")
    #     print(f"Views: {video.views}")
    #     print("---")

    # transcript = get_video_transcript(videos[0].url)
    # print(transcript)
