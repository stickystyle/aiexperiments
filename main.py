import datetime
import os

import environs
import flask
import pirateweather
import requests
from flask import Response
from flask_apscheduler import APScheduler
from icalevents.icalevents import events
from openai import OpenAI

env = environs.Env()
environs.Env.read_env()


app = flask.Flask(__name__)

# initialize scheduler
scheduler = APScheduler()


dir_path = os.path.dirname(os.path.realpath(__file__))

GITHUB_TOKEN = env("GITHUB_TOKEN")
PIRATE_WEATHER_API_KEY = env("PIRATE_WEATHER_API_KEY")
HA_TOKEN = env("HA_TOKEN")
PERSONALITY = env("PERSONALITY")
ICAL_URL = env("ICAL_URL")
HOUR = env("HOUR")
MINUTE = env("MINUTE")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

ha_session = requests.Session()
ha_session.headers.update({"Authorization": f"Bearer {HA_TOKEN}"})


def get_ha_data():
    ha = ha_session.get(
        "http://homeassistant.local:8123/api/states/climate.main_floor",
    )
    main_floor = ha.json()

    ha = ha_session.get(
        "http://homeassistant.local:8123/api/states/zone.home",
    )
    zone_home = ha.json()
    return main_floor, zone_home


def fetch_calendar():
    today = datetime.date.today()
    # The `start`, and `end` arguments of events() do not work as expected, so we filter the results manually
    es = events(ICAL_URL, fix_apple=True, sort=True)
    return [e.summary for e in es if e.start.date() == today]


def build_prompt(main_floor, zone_home):
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

    prompt = f"""Write a good morning message to Christa without using emojis or signing the message, given the following weather conditions for today:
                          Temperature inside the house: {forcast_prompt['Inside Temperature']}
                          Current Conditions: {forcast_prompt['Current Conditions']}
                          Current Temp: {forcast_prompt['Current Temperature']}
                          Conditions for the day: {forcast_prompt['Conditions for Day']}
                          High:  {forcast_prompt['High Temperature']}
                          Low: {forcast_prompt['Low Temperature']}
               And the following items on her calendar for today: {','.join(fetch_calendar())}"""
    app.logger.info(prompt)
    return prompt


def build_response():
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


@scheduler.task(
    "cron", id="write_message", week="*", day_of_week="*", hour=HOUR, minute=MINUTE
)
def _write_message():
    message = build_response()

    with open(f"{dir_path}/message.txt", "w") as f:
        f.write(message)

    return message


@app.route("/write_message")
def write_message():
    message = _write_message()

    return Response(message, mimetype="text/plain")


@app.route("/")
def get_message():
    with open(f"{dir_path}/message.txt", "r") as f:
        return Response(f.read(), mimetype="text/plain")


if __name__ == "__main__":
    scheduler.api_enabled = True
    scheduler.init_app(app)
    scheduler.start()

    app.run(host="0.0.0.0", port=5054)
