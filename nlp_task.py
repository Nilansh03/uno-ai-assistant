import openai
import os
import asyncio

import python_weather


class NlpExpert:
    def __init__(self):
        self.about_uno = 'about_me.txt'
        with open(self.about_uno, 'r') as file:
            file_contents = file.read()
        self.system_role = file_contents
        self.conversation_history = [
            {"role": "system", "content": self.system_role}
        ]
        self.ai_model = "gpt-4o-mini"
        self.explain_text = ['explain', 'elaborate', 'more', 'summary']
        self.weather = ['temperature', 'aqi', 'pollution', 'weather', 'climate']


    def chat_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def ai_chat(self, user_input):
        print(len(self.conversation_history))
        if len(self.conversation_history) > 10:
            self.model_summary()
        self.chat_history("user", user_input)
        ai_response = self.model_calling(user_input)
        self.chat_history("assistant", ai_response)
        return ai_response

    def model_summary(self):
        summary = openai.chat.completions.create(
            model=self.ai_model,  # Use 'gpt-4' or another available model
            messages=[
                {"role": "system",
                 "content": "Summarize the following conversation."},
                *self.conversation_history
            ],
            temperature=0,
            max_tokens=300,
        ).choices[0].message.content

        self.conversation_history = [
            {"role": "system", "content": self.system_role},
            {"role": "assistant", "content": summary}
        ]

    def model_calling(self, user_input):
        if any(i in user_input for i in self.explain_text):
            response = openai.chat.completions.create(
                model=self.ai_model,
                messages=
                    self.conversation_history,

                temperature=0.8,
                max_tokens=300,
            )
            ai_response = response.choices[0].message.content
            return ai_response

        else:
            response = openai.chat.completions.create(
                model=self.ai_model,
                messages=
                    self.conversation_history,
                temperature=0.8,
                max_tokens=300,
                stop=[".", " User:"]
            )
            # print(response)

            ai_response = response.choices[0].message.content

            return ai_response

    def detect_intent(self, user_input):
        for i in self.weather:
            if i in user_input:
                return

    def extract_city_with_gpt(self, input_text):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a bot that extracts city names from text."},
                {"role": "user", "content": f"Extract the city name: '{input_text}'"}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    async def getweather(self, user_input):
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            city = self.extract_city_with_gpt(user_input)
            weather = await client.get(city)

            statement = f"Temperature in {city} is {weather.temperature} degree celsius and the it is {weather.description} outside."
            return statement

    def weather_code(self, user_input):
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        return asyncio.run(self.getweather(user_input))

    def run(self, user_input):
        if any(i in user_input for i in self.weather):
            return self.weather_code(user_input)
        else:
            return self.ai_chat(user_input)



# if __name__ == "__main__":
#
#     nlp_expert = NlpExpert()
#     while True:
#         query = input("You: ")
#
#         if query.lower() == "exit":
#                 print("Goodbye!")
#                 break
#
#         print("Uno: ", str(nlp_expert.run(query)))


