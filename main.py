import pika, time, json, os, sys, dotenv
from service.rabbitmq import RabbitMQ
from service.ai_service_processor import AIProcessor
from langchain_core.output_parsers import JsonOutputParser
from models.user import UserDataForGuidence, AIGuidanceResponse
from utils.utils import init_mongo
from config import settings as global_settings

dotenv.load_dotenv()

def callback(ch, method, properties, body):
        ai_processor = AIProcessor()
        parser = JsonOutputParser(pydantic_object=AIGuidanceResponse)
        user_data = {
            "current_week": 1,
            "chatbot_interaction": "telling chatbot about his depression",
            "emotion_tracking": "Feeling stressful lately"
        }
        user_pass_data = UserDataForGuidence(**user_data)
        res = ai_processor.provide_guidence_process(user_data=user_pass_data, parser=parser)
        print(res, type(res))
        ch.basic_publish(exchange='',
                        routing_key="ai_response",
                        body=json.dumps(res),
                        properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    ))
        print(f"{ch}, {body}, {properties}")

def main():
    print("testing")
    rabbitmq = RabbitMQ()
    print(rabbitmq.channel, rabbitmq.connection)
    rabbitmq.channel.queue_declare("ai_tasks")
    rabbitmq.channel.queue_declare("ai_response")

    print(rabbitmq.__dict__)
    
    try:
        print("Connection to RabbitMQ established successfully.")
        rabbitmq.consume(queue_name='ai_tasks', callback=callback)
        
    except Exception as e:
        print(f"Failed to establish connection to RabbitMQ: {e}")
        sys.exit(1)
    finally:
        rabbitmq.close()
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)