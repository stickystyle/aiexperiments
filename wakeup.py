import environs
import pirateweather
import requests
from openai import OpenAI

env = environs.Env()
environs.Env.read_env()


GITHUB_TOKEN = env("GITHUB_TOKEN")
PIRATE_WEATHER_API_KEY = env("PIRATE_WEATHER_API_KEY")
HA_TOKEN = env("HA_TOKEN")

personalities = {
    "Barbie": "You are Barbie, from the 2023 Barbie movie. A temperature below 70°F is considered cold.",
    "Clueless": "You are Cher from the movie Clueless. A temperature below 70°F is considered cold.",
}

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

    return f"""Write a good morning message to Christa without using emojis or signing the message, given the following weather conditions for today:
                          Temperature inside the house: {forcast_prompt['Inside Temperature']}
                          Current Conditions: {forcast_prompt['Current Conditions']}
                          Current Temp: {forcast_prompt['Current Temperature']}
                          Conditions for the day: {forcast_prompt['Conditions for Day']}
                          High:  {forcast_prompt['High Temperature']}
                          Low: {forcast_prompt['Low Temperature']}"""



def wakeup():
    prompt = build_prompt(*get_ha_data())

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": personalities["Barbie"],
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


def main():
    message = wakeup()

    with open("message.txt", "w") as f:
        f.write(message)

if __name__ == "__main__":
    main()
