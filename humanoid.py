import speech_recognition as sr
import pyttsx3
from nlp_task import NlpExpert
import openai
from camera_task import Camera
import cv2
import threading
import time
custom_pronunciations = {
    "UNO": "ooo-no",
    "ChatGPT": "chat G P T",
    "GPT": "G P T",
    "*": " "
}

exit_command = ["exit", "quit", "close camera", "exit camera"]
nlp_expert = NlpExpert()


def preprocess_text(text):
    for word, replacement in custom_pronunciations.items():
        text = text.replace(word, replacement)
    return text

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    processed_text = preprocess_text(text)  # Apply custom pronunciations
    engine.say(processed_text)
    engine.runAndWait()

def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)
            print("Recognizing...")
            query = recognizer.recognize_google(audio)
            return query
        except sr.WaitTimeoutError:
            print("Listening timeout. Try speaking again.")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("There was an error with the speech recognition service.")
        return None


def detect_command(input_text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a bot that classify anything into either camera task, nlp task or asking to exit. For camera- classify it under person recognition task, new face storing, object detection and room scanning. For NLP task classify it under AI task or weather checking. reply just sub category only, and these category should be: person recognition, new face storing, object detection, room scanning, weather, nlp and exit"},
            {"role": "user", "content": f"Categorize it: '{input_text}'"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def handle_commands(cam1):

    while True:
        print("Say 'exit' to quit.")
        query = listen_to_microphone()
        if query is None:
            continue

        print(f"You: {query}")
        if any(item in query for item in exit_command):
            cam1.running = False
        else:
            detected = detect_command(query)
            print("Processed: ", detected)
            user_command = detected

            if user_command in ["person recognition", "object detection", "new face storing", "room scanning", "exit"] and cam1.running:
                # while cam1.running:
                # user_command = input("Enter task (recognize, detect, quit): ").strip().lower()
                if user_command == "exit":
                    cam1.running = False
                elif user_command in ["person recognition", "object detection"]:
                    cam1.command_queue.put(user_command)
                else:
                    print("Unknown command.")

            else:
                # Only from NLP Task
                response = nlp_expert.run(query)
                print(f"Uno: {response}")
                speak_text(response)


def main():
    cam1 = Camera()
    # Start a thread to handle commands
    command_thread = threading.Thread(target=handle_commands, args=(cam1,), daemon=True)
    command_thread.start()

    current_command = None

    while cam1.running:
        frame = cam1.get_frame()
        if frame is not None:
            if not cam1.command_queue.empty():
                current_command = cam1.command_queue.get()

            if current_command == "person recognition":
                frame = cam1.recognize_person(frame)
            elif current_command == "object detection":
                frame = cam1.detect_objects(frame)

            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam1.release_camera()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     nlp_expert = NlpExpert()
#
#     while True:
#         print("Say 'exit' to quit.")
#         query = listen_to_microphone()
#         if query is None:
#             continue
#
#         print(f"You: {query}")
#
#         detected = detect_command(query)
#         print("Processed: ", detected)
#
#         if detected.lower() == 'person recognition':
#             pass
#
#         elif detected.lower() == 'new face storing':
#             pass
#
#         elif detected.lower() == 'object detection':
#             pass
#
#         if detected.lower() == 'person recognition':
#             pass
#         # if detected.lower() == "exit":
#         #     print("Goodbye!")
#         #     speak_text("Goodbye!")
#         #     break
#
# # '''
# # person recognition, new face storing, object detection, room scanning, weather, nlp and exit
# # '''
#         # Only from NLP Task
#         response = nlp_expert.run(query)
#         print(f"Uno: {response}")
#         speak_text(response)
#
