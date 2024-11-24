import datetime
import logging
from random import randint
from typing import List, Tuple

import environs
import feedparser
import flask
import holidays
import requests
from flask import Response
from homeassistant_api import Client
from icalevents.icalevents import events
from openai import OpenAI

env = environs.Env()
environs.Env.read_env()

logging.basicConfig(level=env("LOG_LEVEL", "INFO"))  # Set the desired logging level


app = flask.Flask(__name__)


client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=env("GITHUB_TOKEN"),
)

ha_client = Client(env("HA_URL"), env("HA_TOKEN"))


def _get_pirate_weather(latitude, longitude) -> str:
    import pirateweather

    forecast = pirateweather.load_forecast(
        key=env("PIRATE_WEATHER_API_KEY"),
        lat=latitude,
        lng=longitude,
    )

    by_hour = forecast.hourly()
    today = forecast.daily().data[0]

    return f"""Current Conditions: {forecast.currently().summary}
              Current Temp: {round(forecast.currently().apparentTemperature)}°F
              Conditions for the day: {by_hour.summary}
              High:  {round(today.d["apparentTemperatureHigh"])}°F
              Low: {round(today.d["apparentTemperatureLow"])}°F"""


def _get_open_weather(latitude, longitude) -> str:
    res = requests.get(
        "https://api.openweathermap.org/data/3.0/onecall",
        params={
            "lat": latitude,
            "lon": longitude,
            "exclude": "minutely,hourly",
            "units": "imperial",
            "appid": env("OPEN_WEATHER_API_KEY"),
        },
    )
    res.raise_for_status()
    data = res.json()
    return f"""Current Conditions: {data['current']['weather'][0]['description']}
          Current Temp: {round(data['current']['feels_like'])}°F
          Conditions for the day: {data['daily'][0]['weather'][0]['description']}
          Morning Temp: {round(data['daily'][0]['temp']['morn'])}°F
          Day Temp: {round(data['daily'][0]['temp']['day'])}°F
          Evening Temp: {round(data['daily'][0]['temp']['eve'])}°F
          Nighttime Temp: {round(data['daily'][0]['temp']['night'])}°F"""


def get_weather() -> str:
    latitude, longitude = get_location_from_ha()
    return _get_open_weather(latitude, longitude)


def get_indoor_temperature() -> int:
    main_floor = ha_client.get_state(entity_id="climate.main_floor")
    return main_floor.attributes["current_temperature"]


def get_location_from_ha() -> Tuple[float, float]:
    zone_home = ha_client.get_state(entity_id="zone.home")
    app.logger.debug("zone_home: %s", zone_home)
    return zone_home.attributes["latitude"], zone_home.attributes["longitude"]


def fetch_calendar() -> str:
    today = datetime.date.today()
    us_holidays = holidays.country_holidays("US")
    # The `start`, and `end` arguments of events() do not work as expected, so we filter the results manually
    es = events(env("ICAL_URL"), fix_apple=True, sort=True)
    cal_events = [e.summary for e in es if e.start.date() == today]
    if today_holidays := us_holidays.get(today):
        cal_events.append(today_holidays)
    app.logger.info("cal_events: %s", cal_events)
    return ", ".join(cal_events).strip()


def get_time_of_day() -> str:
    now = datetime.datetime.now()
    if now.hour < 12:
        return "Morning"
    elif now.hour < 16:
        return "Afternoon"
    elif now.hour < 19:
        return "Evening"
    else:
        return "Night"


def fetch_good_news():
    feed = feedparser.parse("https://www.goodnewsnetwork.org/feed/")
    stories = []
    # Ignore stories with these tags, horoscopes are not good news, and ChatGPT get confused with 'This Day In History'
    ignore_tags = ["Horoscopes", "This Day In History", "On this day"]
    for story in feed.entries:
        tags = [x["term"] for x in story.tags]
        if any(x in tags for x in ignore_tags):
            app.logger.info("skipping story: %s with tags: %s", story.link, tags)
            continue
        stories.append(story)

    app.logger.info("good news: %s", stories[0].link)
    return stories[0].link


def build_prompt():

    try:
        good_news = fetch_good_news()
    except Exception as e:
        app.logger.error("Failed to fetch good news: %s", e)
        good_news = None

    try:
        inside_temperature = get_indoor_temperature()
    except Exception as e:
        app.logger.error("Failed to fetch indoor temperature: %s", e)
        inside_temperature = None

    try:
        weather = get_weather()
    except Exception as e:
        app.logger.error("Failed to fetch weather: %s", e)
        weather = None

    try:
        cal_events = fetch_calendar()
    except Exception as e:
        app.logger.error("Failed to fetch calendar events: %s", e)
        cal_events = None

    prompt = f"Write me a good {get_time_of_day()} message without using emojis or signing the message, given the following weather conditions for today:\n"

    if inside_temperature:
        prompt += f"Temperature inside the house: {inside_temperature}°F\n"

    if weather:
        prompt += f"{weather}\n"

    if cal_events:
        prompt += f"And Today's calendar events: {cal_events}\n"

    if good_news:
        prompt += f"Finally, here's a link to some good news that I'd like you to summarize: {good_news}\n"
    app.logger.info(prompt)
    return prompt


def build_response() -> str:
    prompt = build_prompt()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": env("PERSONALITY"),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=env.float("TEMPERATURE", 0.0),
        top_p=1.0,
        max_tokens=1000,
        model=env("MODEL_NAME"),
    )

    return response.choices[0].message.content


@app.route("/")
def get_message() -> Response:
    message = build_response()
    app.logger.info("message: %s", message)
    return Response(message, mimetype="text/plain")


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5054)
