import datetime
import os
import logging
from typing import List, Tuple

import environs
import flask
import pirateweather
import requests
from flask import Response
from icalevents.icalevents import events
from openai import OpenAI
import holidays

env = environs.Env()
environs.Env.read_env()

logging.basicConfig(level=env("LOG_LEVEL", "INFO"))  # Set the desired logging level


app = flask.Flask(__name__)


GITHUB_TOKEN = env("GITHUB_TOKEN")
PIRATE_WEATHER_API_KEY = env("PIRATE_WEATHER_API_KEY")
HA_TOKEN = env("HA_TOKEN")
PERSONALITY = env("PERSONALITY")
ICAL_URL = env("ICAL_URL")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

ha_session = requests.Session()
ha_session.headers.update({"Authorization": f"Bearer {HA_TOKEN}"})


def get_ha_data() -> Tuple[dict, dict]:
    ha = ha_session.get(
        "http://homeassistant.local:8123/api/states/climate.main_floor",
    )
    main_floor = ha.json()
    app.logger.debug("main_floor: %s", main_floor)

    ha = ha_session.get(
        "http://homeassistant.local:8123/api/states/zone.home",
    )
    zone_home = ha.json()
    app.logger.debug("zone_home: %s", zone_home)
    return main_floor, zone_home


def fetch_calendar() -> List[str]:
    today = datetime.date.today()
    us_holidays = holidays.country_holidays('US')
    # The `start`, and `end` arguments of events() do not work as expected, so we filter the results manually
    es = events(ICAL_URL, fix_apple=True, sort=True)
    cal_events = [e.summary for e in es if e.start.date() == today]
    if today_holidays := us_holidays.get(today):
        cal_events.append(today_holidays)
    app.logger.info("cal_events: %s", cal_events)
    return cal_events


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


def build_prompt(main_floor: dict, zone_home: dict):
    forecast = pirateweather.load_forecast(
        PIRATE_WEATHER_API_KEY,
        zone_home["attributes"]["latitude"],
        zone_home["attributes"]["longitude"],
    )

    by_hour = forecast.hourly()
    today = forecast.daily().data[0]
    forcast_prompt = {
        "Inside Temperature": f"{main_floor['attributes']['current_temperature']}°F",
        "Current Conditions": forecast.currently().summary,
        "Current Temperature": f"{round(forecast.currently().apparentTemperature)}°F",
        "Conditions for Day": by_hour.summary,
        "High Temperature": f"{round(today.d['apparentTemperatureHigh'])}°F",
        "Low Temperature": f"{round(today.d['apparentTemperatureLow'])}°F",
    }
    cal_events = fetch_calendar()
    prompt = f"""Write me a good {get_time_of_day()} message without using emojis or signing the message, given the following weather conditions for today:
                          Temperature inside the house: {forcast_prompt['Inside Temperature']}
                          Current Conditions: {forcast_prompt['Current Conditions']}
                          Current Temp: {forcast_prompt['Current Temperature']}
                          Conditions for the day: {forcast_prompt['Conditions for Day']}
                          High:  {forcast_prompt['High Temperature']}
                          Low: {forcast_prompt['Low Temperature']}"""
    if cal_events:
        prompt += f"\n\nAnd Today's calendar events: {', '.join(cal_events)}"
    app.logger.info(prompt)
    return prompt


def build_response() -> str:
    prompt = build_prompt(*get_ha_data())

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": PERSONALITY,
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


def _write_message() -> str:
    message = build_response()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f"{dir_path}/message.txt", "w") as f:
        f.write(message)

    return message


@app.route("/")
def get_message() -> Response:
    message = build_response()
    app.logger.info("message: %s", message)
    return Response(message, mimetype="text/plain")


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5054)
