#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1. Write a python program to extract the video URL of the first five videos.
To extract video URLs from a webpage, you can use a combination of libraries like requests to fetch the webpage content and BeautifulSoup to parse the HTML. In this example, I'll demonstrate how to extract video URLs from a YouTube search page for the first five videos:
import requests
from bs4 import BeautifulSoup

def extract_video_urls(search_query, num_videos=5):
    base_url = f"https://www.youtube.com/results"
    params = {
        "search_query": search_query,
        "sp": "EgIQAQ%253D%253D",  # This parameter is used for filtering video results
    }

    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    video_urls = []
    for link in soup.find_all("a", {"class": "yt-simple-endpoint", "aria-hidden": "true"}):
        video_id = link["href"].replace("/watch?v=", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_urls.append(video_url)

        if len(video_urls) >= num_videos:
            break

    return video_urls

if __name__ == "__main__":
    search_query = input("Enter the search query: ")
    video_urls = extract_video_urls(search_query, num_videos=5)

    print("Extracted Video URLs:")
    for i, url in enumerate(video_urls, start=1):
        print(f"{i}. {url}")
In this example, you input a search query, and the program will extract the video URLs of the first five videos from the YouTube search results page using the provided libraries.

Remember that web scraping can be subject to changes in website structures, and YouTube might employ measures to prevent or limit scraping. Always ensure you're following the website's terms of service and guidelines when accessing their content programmatically.
# In[ ]:





# Q2. Write a python program to extract the URL of the video thumbnails of the first five videos.
To extract the URLs of video thumbnails from a YouTube search page for the first five videos, you can modify the previous program to target the thumbnail elements and extract the data-thumb attribute or similar attributes. Here's how you can do it:
import requests
from bs4 import BeautifulSoup

def extract_thumbnail_urls(search_query, num_videos=5):
    base_url = f"https://www.youtube.com/results"
    params = {
        "search_query": search_query,
        "sp": "EgIQAQ%253D%253D",  # This parameter is used for filtering video results
    }

    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    thumbnail_urls = []
    for thumbnail in soup.find_all("img", {"class": "style-scope yt-img-shadow"}):
        if "data-thumb" in thumbnail.attrs:
            thumbnail_url = thumbnail["data-thumb"]
            thumbnail_urls.append(thumbnail_url)

        if len(thumbnail_urls) >= num_videos:
            break

    return thumbnail_urls

if __name__ == "__main__":
    search_query = input("Enter the search query: ")
    thumbnail_urls = extract_thumbnail_urls(search_query, num_videos=5)

    print("Extracted Thumbnail URLs:")
    for i, url in enumerate(thumbnail_urls, start=1):
        print(f"{i}. {url}")
In this program, we are searching for img elements with the class style-scope yt-img-shadow, which typically contain video thumbnail URLs. We extract the data-thumb attribute from these elements. Just like before, input a search query, and the program will extract the thumbnail URLs of the first five videos from the YouTube search results page.

Please note that YouTube's website structure may change over time, which could affect the way you need to extract thumbnail URLs. Make sure to review YouTube's terms of service and guidelines when accessing their content programmatically.
# In[ ]:





# Q3. Write a python program to extract the title of the first five videos.
To extract the titles of the first five videos from a YouTube search page, you can continue building on the previous program. Here's how you can do it:
import requests
from bs4 import BeautifulSoup

def extract_video_titles(search_query, num_videos=5):
    base_url = f"https://www.youtube.com/results"
    params = {
        "search_query": search_query,
        "sp": "EgIQAQ%253D%253D",  # This parameter is used for filtering video results
    }

    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    video_titles = []
    for title in soup.find_all("a", {"class": "yt-simple-endpoint style-scope ytd-video-renderer"}):
        if title.has_attr("title"):
            video_title = title["title"]
            video_titles.append(video_title)

        if len(video_titles) >= num_videos:
            break

    return video_titles

if __name__ == "__main__":
    search_query = input("Enter the search query: ")
    video_titles = extract_video_titles(search_query, num_videos=5)

    print("Extracted Video Titles:")
    for i, title in enumerate(video_titles, start=1):
        print(f"{i}. {title}")
In this program, we are searching for a elements with the class yt-simple-endpoint style-scope ytd-video-renderer, which typically contain the video titles as the title attribute. We extract this attribute to get the video titles. As before, input a search query, and the program will extract the titles of the first five videos from the YouTube search results page.

Remember that YouTube's website structure might change, so adapt the program accordingly if needed. Always make sure to respect YouTube's terms of service and guidelines when accessing their content programmatically.
# In[ ]:





# Q4. Why is flask used in this Web Scraping project?
To extract the number of views of the first five videos from a YouTube search page, you can follow a similar approach as in the previous programs. Here's how you can do it:
import requests
from bs4 import BeautifulSoup

def extract_video_views(search_query, num_videos=5):
    base_url = f"https://www.youtube.com/results"
    params = {
        "search_query": search_query,
        "sp": "EgIQAQ%253D%253D",  # This parameter is used for filtering video results
    }

    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    video_views = []
    for views in soup.find_all("span", {"class": "style-scope ytd-video-meta-block"}):
        if "views" in views.get_text():
            video_view_count = views.get_text().split()[0]
            video_views.append(video_view_count)

        if len(video_views) >= num_videos:
            break

    return video_views

if __name__ == "__main__":
    search_query = input("Enter the search query: ")
    video_views = extract_video_views(search_query, num_videos=5)

    print("Number of Views for First Five Videos:")
    for i, views in enumerate(video_views, start=1):
        print(f"{i}. {views} views")
In this program, we are searching for span elements with the class style-scope ytd-video-meta-block, which usually contains the information about the number of views. We then parse the text within the span element to extract the view count. As with previous examples, input a search query, and the program will extract the number of views of the first five videos from the YouTube search results page.

As always, be aware that YouTube's website structure can change over time, so you might need to adjust the program accordingly. Also, remember to follow YouTube's terms of service and guidelines when accessing their content programmatically.
# In[ ]:





# Q5. Write the names of AWS services used in this project. Also, explain the use of each service.
To extract the time of posting of the first five videos from a YouTube search page, you can use a similar approach as before. However, extracting the exact time of posting can be more complex as YouTube's UI often represents this information differently. Here's a general example of how you might extract this information
import requests
from bs4 import BeautifulSoup

def extract_video_posting_time(search_query, num_videos=5):
    base_url = f"https://www.youtube.com/results"
    params = {
        "search_query": search_query,
        "sp": "EgIQAQ%253D%253D",  # This parameter is used for filtering video results
    }

    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    video_posting_times = []
    for time_element in soup.find_all("span", {"class": "style-scope ytd-video-meta-block"}):
        if "ago" in time_element.get_text():
            video_posting_time = time_element.get_text().strip()
            video_posting_times.append(video_posting_time)

        if len(video_posting_times) >= num_videos:
            break

    return video_posting_times

if __name__ == "__main__":
    search_query = input("Enter the search query: ")
    video_posting_times = extract_video_posting_time(search_query, num_videos=5)

    print("Posting Times for First Five Videos:")
    for i, posting_time in enumerate(video_posting_times, start=1):
        print(f"{i}. {posting_time}")
In this example, we're looking for span elements with the class style-scope ytd-video-meta-block, which might contain information about the time of posting. However, this method might not work perfectly for all cases since YouTube's website structure can vary.

You may need to adjust the program based on how YouTube presents the posting time information. Some videos might show "ago" with a relative timestamp, while others might have more precise timestamps or different formatting. This program provides a starting point, but you might need to experiment and adapt based on YouTube's specific layout.

Always be mindful of potential changes to YouTube's website structure and follow their terms of service when scraping content.
# #  <P style="color:GREEN"> Thank You ,That's All </p>
